int[] x_coordinates = {3, 5, 4, 5, 2, 7, 1, 8};
int[] y_coordinates = {1, 1, 2, 2, 5, 4, 0, 0};

double[] xy1 = {4.333333, 1};
double[] xy2 = {4.5, 4.5};

double[] cluster1 = new double[x_coordinates.length];
double[] cluster2 = new double[x_coordinates.length];
double variance = 0;

for (int i = 0; i < x_coordinates.length; i++) {
    double distanceC1 = Math.sqrt(Math.pow(xy1[0] - x_coordinates[i], 2) + Math.pow(xy1[1] - y_coordinates[i], 2));
    System.out.println("A -> Centroid 1: " + distanceC1);
    double distanceC2 = Math.sqrt(Math.pow(xy2[0] - x_coordinates[i], 2) + Math.pow(xy2[1] - y_coordinates[i], 2));

    if (distanceC1 < distanceC2) {
        cluster1[i] = distanceC1;
    } else if (distanceC1 > distanceC2) {
        cluster2[i] = distanceC2;
    }
}

System.out.println();

for (int i = 0; i < cluster1.length; i++) {
    variance = variance + Math.pow(cluster1[i], 2);
}for (int i = 0; i < cluster2.length; i++) {
    variance = variance + Math.pow(cluster2[i], 2);
}

for (int i = 0; i < x_coordinates.length; i++) {
    double distanceC2 = Math.sqrt(Math.pow(xy2[0] - x_coordinates[i], 2) + Math.pow(xy2[1] - y_coordinates[i], 2));
    System.out.println("B -> Centroid 2: " + distanceC2);
}

System.out.println("The within cluster variance is: " + variance);