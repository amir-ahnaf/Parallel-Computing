#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>
// -----------------------------------------------------------------------------------------
// Function to find the prefix scan coefficient via parallel computing
// -----------------------------------------------------------------------------------------

// Main Program to obtain number of random number generated, executing prefix_scan via
// parallel computing, and displaying the prefix scan coefficient

// Input argument: number of core provided, number of random number wanted

// Methodology: Up and Down Phase

// Output: prefix scan coefficient

// Compilation: mpicc <file_name>: mpicc prefix.cpp

// Execution: mpiexec <command> <: mpiexec -nearest_power 4 ./a

// Required header files: stdio.h (for printf), time.h (for time),
//                        stdlib.h (for srand, free), mpi.h (for MPI_Bcast)

// ----------------------------------------------------------------------------------------


// nearestPowerOf2() function finds and returns the nearest power of 2 for N
int nearestPowerOf2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// generateRandomNumbers function generate N number of random number in random_numbers
void generateRandomNumbers(int n, int* random_numbers) {
    // random the time to prevent repetitive number for each execution
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        random_numbers[i] = rand() % 10 + 1;
    }
}

// ----------------------------------------------------------------------------------------
int main(int argc, char *argv[]) {

// This code initiate the MPI, get user random_number, generate random numbers, broadcast it other cores
    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int n, nearest_power, *random_numbers;

    // Part 1 - Initialisation
    // Process 0 to obtain N from the user and generate N random numbers in random_numbers
    // nearest_power is the nearest power of 2 of N, used to generate array, and padding
    if (rank == 0) {
        printf("Prefix Scan Coefficient generator \n");
        printf("[%d] Enter the number of elements: ", rank);
        fflush(stdout);
        scanf("%d", &n);

        // Start time recording
        start_time = MPI_Wtime();

        nearest_power = nearestPowerOf2(n);
        random_numbers = (int*)malloc(nearest_power * sizeof(int));
        // generate N random number in nearest_power sized array
        generateRandomNumbers(n, random_numbers);
        printf("[%d] Generated %d random numbers: ", rank, n);
        for (int i = 0; i < n; i++) {
            printf("%d ", random_numbers[i]);
        }
        printf("\n");

        // Padding the excess elements in the array with 0
        if (nearest_power > n) {
            for (int j = n; j < nearest_power; j++) random_numbers[j] = 0;
        }
    }

    // -------------------------------------------------------------------

    // Part 2 - Broadcasting
    // Broadcasting n to all core, find the nearest_power, memory allocation and broadcasting for randomNumber

    // Broadcast n to all processes,
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank != 0) {
        nearest_power = nearestPowerOf2(n);
        random_numbers = (int*)malloc(nearest_power * sizeof(int));
    }
    // Broadcast the randomNumber generated in process 0 to other cores
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(random_numbers, nearest_power, MPI_INT, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------


    // Part 3 - Prefix scan coefficient
    // In this prefix scan, firstly the random numbers are divided to each process by size for each processes
    // Up phase is executed, and compensated since there is error during the core division
    // In the phases, the interval is distance between the elements in the array
    // i is the targeted element in for each interval, right is index for element in right side,
    // left is index for element in left side which going to be added into

    // Variable initialisation
    int process_size = nearest_power / size;
    int begin_idx = rank * process_size;
    int end_idx =  (rank + 1) * process_size;
    int *local_sum = (int*) malloc(process_size * sizeof(int));
    int *gathered_sums = (int*) malloc(size * sizeof(int));
    int *prefix = (int*) malloc(nearest_power * sizeof(int));

    // Up phase with prefix sum for compensation
    for (int interval = 1; interval < nearest_power; interval *= 2) {
        for (int i = 0; i < nearest_power; i += 2 * interval) {
            int left = i + interval - 1;
            int right = i + 2 * interval - 1;
            if (right < nearest_power) {
                prefix[right] += prefix[left];
            }
        }
    }

    // Up phase compensation
    int previous_value = 0;
    for (int i = begin_idx; i < end_idx; i++) {
        previous_value += random_numbers[i];
        local_sum[i - begin_idx] = previous_value;
    }
    // Gather the last value of local_sum from each process in the root.
    MPI_Gather(&local_sum[process_size - 1], 1, MPI_INT, gathered_sums, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Down Phase
    for (int interval = nearest_power / 2; interval > 0; interval /= 2) {
        for (int i = 0; i < nearest_power; i += 2 * interval) {
            int left = i + interval - 1;
            int right = i + 2 * interval - 1;
            if (right < nearest_power) {
                prefix[right] += prefix[left];
            }
        }
    }

    // Down phase compensation
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            gathered_sums[i] += gathered_sums[i - 1];
        }
    }

    // Broadcast the accumulated sums to all processes.
    MPI_Bcast(gathered_sums, size, MPI_INT, 0, MPI_COMM_WORLD);
    // Adjust local_sum for each process
    if (rank > 0) {
        int offset = gathered_sums[rank - 1];
        for (int i = 0; i < process_size; i++) {
            local_sum[i] += offset;
        }
    }
    // Gather the adjusted local sums to the root process for the final result.
    MPI_Gather(local_sum, process_size, MPI_INT, random_numbers, process_size, MPI_INT, 0, MPI_COMM_WORLD);


    // Display the calculated prefix scan coefficient, free the memory
    if (rank == 0) {
        printf("[%d] Prefix scan coefficients: ", rank);
        for (int i = 0; i < n; i++) {
            printf("%d ", random_numbers[i]);
        }
        printf("\n");

        free(gathered_sums);
        free(random_numbers);
    } else {
        free(random_numbers);
    }
    free(local_sum);

    // Finish time recording, root displays the time
    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
