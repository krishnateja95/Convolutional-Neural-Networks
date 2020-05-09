#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1

void gettimeofday(time_t *tp, char *_)
    {
    *tp = clock();
    return;
    }

double get_seconds(time_t timeStart, time_t timeEnd)
{
    return (double)(timeEnd - timeStart) / CLOCKS_PER_SEC;
}


#else

double get_seconds(struct timeval timeStart, struct timeval timeEnd)
    {
    return ((timeEnd.tv_sec - timeStart.tv_sec) * 1000000 + timeEnd.tv_usec - timeStart.tv_usec) / 1.e6;
    }

#endif

#define SIZE 40

float ****image;


int weights_shape[3][4] = 
{ 
    { 32,   3,  5, 5 },
    { 32,  32,  5, 5 },
    { 64,  32,  5, 5 }
};

float *****conv;

int activation_shape[3] = {64, SIZE, SIZE};
float ****activation_memory_1;
float ****activation_memory_2;

void clear_memory(float ****memory, int batch_size)
    {
    int i, j, k, m;
    
    for (m = 0; m < batch_size; m++)
        {
        for (i = 0; i < activation_shape[0]; i++)
            {
            for (j = 0; j < activation_shape[1]; j++)
                {
                for (k = 0; k < activation_shape[2]; k++)
                    {
                    memory[m][i][j][k] = 0.0;
                    }
                }
            }
        }
    }





void initialize_memory(int batch_size)
    {
    int i, j, k, l;
        
    image = malloc(batch_size * sizeof(float***));
    
    for (i = 0; i < batch_size; i++)
        {
        image[i] = malloc(3 * sizeof(float**));
        
        for(j = 0; j < 3; j++)
            {
            image[i][j] = malloc(SIZE * sizeof(float*));
            
            for(k = 0; k < SIZE; k++)
                {
                image[i][j][k] = malloc(SIZE * sizeof(float));
                }
            }
        }
    

    conv = malloc(3 * sizeof(float****));

    for (l = 0; l < 3; l++)
        {
        conv[l] = malloc(weights_shape[l][0] * sizeof(float***));
        for (i = 0; i < weights_shape[l][0]; i++)
            {
            conv[l][i] = malloc(weights_shape[l][1] * sizeof(float**));
            for (j = 0; j < weights_shape[l][1]; j++)
                {
                conv[l][i][j] = malloc(weights_shape[l][2] * sizeof(float*));
                for (k = 0; k < weights_shape[l][2]; k++)
                    {
                    conv[l][i][j][k] = malloc(weights_shape[l][3] * sizeof(float));
                    }
                }
            }
        }
        
    
    // Init mem_blocks
        
    activation_memory_1 = malloc(batch_size * sizeof(float***));
    activation_memory_2 = malloc(batch_size * sizeof(float***));
    
    for(i = 0; i < batch_size; i++)
        {
        activation_memory_1[i] = malloc(activation_shape[0] * sizeof(float**));        
        activation_memory_2[i] = malloc(activation_shape[0] * sizeof(float**));
                
        for(j = 0; j < activation_shape[0]; j++)
            {
            activation_memory_1[i][j] = malloc(activation_shape[1] * sizeof(float*));
            
            activation_memory_2[i][j] = malloc(activation_shape[1] * sizeof(float*));
            
            
            for(k = 0; k < activation_shape[1]; k++)
                {
                activation_memory_1[i][j][k] = malloc(activation_shape[2] * sizeof(float));
                activation_memory_2[i][j][k] = malloc(activation_shape[2] * sizeof(float));
                }
            }
        }
                   
    
    clear_memory(activation_memory_1, batch_size);
    clear_memory(activation_memory_2, batch_size);
                
}
                

void image_weight_random(int batch_size)
    {
    
    int i,j,k,l,m;
    
    for (i = 0; i < batch_size; i++)
        {
        for(j = 0; j < 3; j++)
            {
            for(k = 0; k < SIZE; k++)
                {
                for(l = 0; l < SIZE; l++)
                    {
                    image[i][j][k][l] = rand()%256;;
                    }
                }
            }
        }
    
    
    
    
    for (l = 0; l < 3; l++)
        {
        for (i = 0; i < weights_shape[l][0]; i++)
            {
            for (j = 0; j < weights_shape[l][1]; j++)
                {
                for (k = 0; k < weights_shape[l][2]; k++)
                    {
                    for (m = 0; m < weights_shape[l][2]; m++)
                        {
                        conv[l][i][j][k][m] = rand()%256;;
                        }                            
                    }
                }
            }
        }
   }


void free_memory(int batch_size)
{
    int i, j, k, l;

    // Free image memory
    
    for(k = 0; k < batch_size; k++)
        {
        for (i = 0; i < 3; i++)
            {
            for (j = 0; j < SIZE; j++)
                {
                free(image[k][i][j]);
                }
            free(image[k][i]);
            }
        free(image[k]);
    }
    free(image);
    
    
    
    // Free convolution weights
    for (l = 0; l < 3; l++) {
        for (i = 0; i < weights_shape[l][0]; i++) {
            for (j = 0; j < weights_shape[l][1]; j++) {
                for (k = 0; k < weights_shape[l][2]; k++) {
                    free(conv[l][i][j][k]);
                }
                free(conv[l][i][j]);
            }
            free(conv[l][i]);
        }
        free(conv[l]);
    }
    free(conv);
    
    
    for(k = 0; k < batch_size; k++)
        {
        for (i = 0; i < activation_shape[0]; i++) 
            {
            for (j = 0; j < activation_shape[1]; j++) 
                {
                free(activation_memory_1[k][i][j]);
                free(activation_memory_2[k][i][j]);
                
            }
            free(activation_memory_1[k][i]);
            free(activation_memory_2[k][i]);
            }
        free(activation_memory_1[k]);
        free(activation_memory_2[k]);
        }
    
    
    free(activation_memory_1);
    free(activation_memory_2);
    
}


void add_bias_and_relu(float **out, int size)
    {
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            out[i][j] = (out[i][j] < 0) ? 0 : out[i][j];
        }
    }
}

void convolution(float ****output_matrix, float ****input_matrix, 
                       int layer, int activation_size, int batch_size, int stride)
    {
        
    int sum;
    int i,j,m,k,w, e,r;
    
    for(w = 0; w < batch_size; w++)
        {
        for(i = 0; i < weights_shape[layer][0]; i++)
            {
            for (j = 0; j < weights_shape[layer][1]; j++)
                {
                for (k = 0; k < activation_size; k = k + stride)
                    {
                    for (m = 0; m < activation_size; m = m + stride)
                        {
                        sum = 0;
                        for (e = 0; e < weights_shape[layer][2]; e++)
                            {
                            for(r = 0; r < weights_shape[layer][3]; r++)
                                {
                                sum += input_matrix[w][j][k + e][m + r] * conv[layer][i][j][e][r];
                                }
                             }
                        output_matrix[w][i][k][m] = output_matrix[w][i][k][m] + sum;
                        }
                    }
                add_bias_and_relu(output_matrix[w][i], activation_size);
                }
            }
        }
    }






float max_of_4(float a, float b, float c, float d) {
    if (a >= b && a >= c && a >= d) {
        return a;
    }
    if (b >= c && b >= d) {
        return b;
    }
    if (c >= d) {
        return c;
    }
    return d;
}


float max_of_2(float a, float b) {
    if (a >= b )
        {
        return a;
        }    
    else
        return b;
}


void maxpooling(float ****out, int size, int batch_size, int layer, int kernel, int stride) 
    {
    int i, j, k, m,x,z;
    float max; 
    for(k = 0; k < batch_size; k++)
        {
        for (m = 0; m < weights_shape[layer][1]; m++)
            {
            for (i = 0; i < size; i+=stride)
                {
                for (j = 0; j < size; j+=stride)
                    {
                    max = 127;
                    for(x = 0; x < kernel; x++)
                        {
                        for(z = 0; z < kernel; z++)
                            {
                            max = max_of_2(out[k][m][i + x][j + z], max);
                            }
                        }
                        out[k][m][i / 2][j / 2] = max;
                    }
                }
            }
        }
    }


void ConvNet(int batch_size) 

{
    clear_memory(activation_memory_1, batch_size);
    clear_memory(activation_memory_2, batch_size);
    
    int i, j;
    int layer, activation_size, stride, kernel;

    
    layer = 0;
    activation_size = 36;
    stride = 1;
    convolution(activation_memory_1, image, layer, activation_size, batch_size, stride);
    
    activation_size = 34;
    stride = 2;
    kernel = 3;
    maxpooling(activation_memory_1, activation_size, batch_size, layer, kernel, stride);
    
    
    layer = 1;
    activation_size = 18;
    stride = 1;
    convolution(activation_memory_2, activation_memory_1, layer, activation_size, batch_size, stride);
    clear_memory(activation_memory_1, batch_size);
    
    
    activation_size = 18;
    stride = 2;
    kernel = 3;
    maxpooling(activation_memory_2, activation_size, batch_size, layer, kernel, stride);
    
    
    layer = 2;
    activation_size = 12;
    stride = 1;
    convolution(activation_memory_1, activation_memory_2, layer, activation_size, batch_size, stride);
    clear_memory(activation_memory_2, batch_size);
    
    activation_size = 10;
    stride = 2;
    kernel = 3;
    maxpooling(activation_memory_1, activation_size, batch_size, layer, kernel, stride);

    return;
}

int main(int argc, char *argv[])    
    {
        
   
        
    #ifndef _WIN32
    struct timeval timeStart, timeEnd;
    #else
    time_t timeStart, timeEnd;
    #endif
    double deltaTime;
    
    int iter, batch_iter;
    int num_iter = 2;
    int batch_size[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32,64};
        
    for(batch_iter = 0; batch_iter < 17; batch_iter++)
    {
    
    gettimeofday(&timeStart, NULL);
    
    for(iter = 0; iter < num_iter; iter++)
        {
        
        initialize_memory(batch_size[batch_iter]);
        image_weight_random(batch_size[batch_iter]);

        ConvNet(batch_size[batch_iter]);

        free_memory(batch_size[batch_iter]);    
        }

        
    gettimeofday(&timeEnd, NULL);
    deltaTime = get_seconds(timeStart, timeEnd);
    printf("Average Infer image float: %.3lf sec\n", deltaTime/num_iter);
    
    FILE * fp;
    fp = fopen ("Results.txt","a");
    fprintf (fp, "\n");
    fprintf (fp, "Image and Kernel Reuse\n");        
    fprintf (fp, "Batch Size = %d \n",batch_size[batch_iter]);
    fprintf (fp, "Average Infer time = %.3lf \n",deltaTime/num_iter);
    fprintf (fp, "\n");
    fprintf (fp, "---------------------------------------------------------------------------------------------");
    fprintf (fp, "\n");
    
    
    fclose (fp);
        
    }
    
    return 0;
}


