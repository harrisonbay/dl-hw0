#include <assert.h>
#include <math.h>
#include "uwnet.h"

#define ALPHA 0.1
// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    
    int epsilon = 5;  // subtracting constant from exp to make sure no overflow
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            //double x = m.data[i*m.cols + j];
            if(a == LOGISTIC) {
                m.data[i*m.cols + j] = 1/(1 + exp(-1*m.data[i*m.cols + j]));

            } else if (a == RELU) {
                if (m.data[i*m.cols + j] < 0) {
                  m.data[i*m.cols + j] = 0;
                }

            } else if (a == LRELU) {
                if (m.data[i*m.cols + j] < 0) {
                  m.data[i*m.cols + j] *= ALPHA;
                }
                
            } else if (a == SOFTMAX) {
                m.data[i*m.cols + j] = exp(m.data[i*m.cols + j] - epsilon);
            }

            sum += m.data[i*m.cols + j];
        }

        if (a == SOFTMAX) {
            for (int k = 0; k < m.cols; k++) {
              m.data[i*m.cols + k] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;

    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            if (a == LOGISTIC) {
              d.data[i*d.cols + j] *= x * (1 - x);
            } else if (a == RELU) {
                if (x < 0) {
                  d.data[i*d.cols + j] = 0;
                }
            } else if (a == LRELU) {
                if (x < 0) {
                  d.data[i*d.cols + j] *= ALPHA;
                }
            }
        }
    }
}
