#include <stdlib.h>
#include <stdio.h>

#include "helper.h"

/* can be used to plot loss and accuracy */
void plot_metrics(double *epochs, double *training_metrics, double *val_metrics, int num_epochs, char *title, char *ylabel, char *filename) {
    char setTitle[100];
    char setYLabel[100];
    char setFilename[100];
    sprintf(setTitle, "set title \"%s\"", title);
    sprintf(setYLabel, "set ylabel \"%s\"", ylabel);
    sprintf(setFilename, "set output \"%s\"", filename);
    char * basicCmds[] = {setTitle, "set xlabel \"Number of Epochs\"", setYLabel, "set terminal png", setFilename};
    FILE * temp = fopen("data.temp", "w");
    /*Opens an interface that one can use to send commands as if they were typing into the
     *     gnuplot command line.  "The -persistent" keeps the plot open even after your
     *     C program terminates.
     */
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (int i=0; i < 5; i++)
    {
        fprintf(gnuplotPipe, "%s \n", basicCmds[i]); //Send commands to gnuplot one by one.
    }
    if (val_metrics == NULL) {
        for (int i=0; i < num_epochs; i++)
        {
            fprintf(temp, "%lf %lf \n", epochs[i], training_metrics[i]); //Write the data to a temporary file
        }
        fprintf(gnuplotPipe, "plot 'data.temp' title 'Training' with lines \n");
    } else {
        for (int i=0; i < num_epochs; i++)
        {
            fprintf(temp, "%lf %lf %lf \n", epochs[i], training_metrics[i], val_metrics[i]); //Write the data to a temporary file
        }
        fprintf(gnuplotPipe, "plot \"data.temp\" using 1:2 title 'Training' with lines, \"data.temp\" using 1:3 title 'Validation' with lines \n");
        // fprintf(gnuplotPipe, "plot \"data.temp\" using 1:3 title 'Validation' with lines \n");
    }

    fclose(temp);
    pclose(gnuplotPipe);
}