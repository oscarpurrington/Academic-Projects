"""
Various imports are used in this program.
Numpy to handle numerical caluclations,
MatPlotLib for graphing.SKLearn is the principle codebase used to perform
the analysis on the data, including the Linear regression methods, the train-
test split, the R squared values, and the scaler used for the Stochastic
Gradient Descent method (from SKLearn also).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def data_read_clean(file_name):
    """
    Reads the given freefall data, removes
    invalid rows, and extracts information.

    Method
    ------
    Invalid rows are defined as rows missing values, containing NaN values
    where a number should be, containing negative values (all values are
    positive) or rows containing values that are not permitted.

    The line initial is checked and if it is a hashtag the line is known to be
    in the header. These lines are checked to extract the features, materials,
    and units, and these values are stripped and put into their own arrays.
    This method is preferred over a lookup table to provide expandability.

    For actual data entries, the whole line is split into an array, the length
    of the array is checked to look for missing entries, the initial is checked
    against the materials array to check for invalid materials, and the data
    is then modified to remove any characters such as ., E, etc. and then
    checked to be a non-negative integer. If the line satisfies every condition
    it is appeneded into the cleaned_array list (without the material), the
    density alone is enough identifier.


    Parameters
    ----------
    file_name : string
        Path to freefall data CSV file

    Returns
    -------
    cleaned_array : list
        A list of lists containing only valid rows
    materials : list
        A list of experimental materials
    features : list
        A list of the Physical features
    units : list
        The units for each feature
    """

    with open(file_name) as f:  # opens CSV file

        cleaned_array = []  # list initialisations
        materials = []
        features = []
        units = []
        bad_lines = 0
        total_lines = 0

        for line in f:  # loops each entry
            total_lines += 1
            if line[0] == "#":  # condition to see if entry in metadata
                if "Material types" in line:
                    materials = (
                        f.readline().lower().strip().strip("#").split()
                    )  # removes extra characters
                    continue
                elif "Features" in line:
                    features = (
                        f.readline()
                        .lower()
                        .strip()
                        .strip("#")
                        .replace(",", "")
                        .split()
                    )
                    continue
                elif "Units" in line:
                    units = f.readline().strip().strip("#").split()
                    continue

            else:
                formatted_line = [
                    entry.strip()
                    # format entry
                    for entry in line.lower().split(",")
                ]

                # if incorrect material - remove
                if formatted_line[0] not in materials:
                    bad_lines += 1
                    continue
                elif len(formatted_line) != 8:  # entry missing a value?
                    bad_lines += 1
                    continue
                elif any(
                    not value.replace(".", "")
                    .replace("e", "")
                    .replace("-", "")
                    .replace("+", "")
                    .isdigit()
                    or float(value) <= 0  # entry negative or NaN?
                    for value in formatted_line[1:]
                ):
                    bad_lines += 1
                    continue
                else:
                    cleaned_array.append(formatted_line[1:])  # append entry
    print(
        f"""No. of lines removed: {bad_lines}/{total_lines}
        ({(bad_lines/total_lines * 100):.2f})% of total"""
    )
    return cleaned_array, materials, features, units


def display_statistics(cleaned_array, materials, features, units):
    """
    Displays some numerical statistics of the cleaned data
    (max, min, mean, s.dev for each feature)

    Method
    ------
    An intial topline is printed, then some default python and numpy functions
    are run on each feature (column). These are max(), min(), np.mean() and
    np.std(). These are then printed with their features and units in a clean
    table.

    Parameters
    ----------
    As in data_read_clean().

    Returns
    -------
    None.

    """
    print(
        f"{'Feature':<20} | {'Max':<10} | {'Min':<10} | {
          'Mean':<10} | {'Standard Deviation':<10}"
    )
    for i in range(7):
        column = []
        for row in cleaned_array:

            column.append(float(row[i]))

        feature_max = max(column)
        feature_min = min(column)
        feature_mean = np.mean(column)
        feature_std = np.std(column)
        print(
            f"{features[i+1]:<12} {units[i+2]:<8} {feature_max:<12} {
              feature_min:<12} {feature_mean:<12.2f} {feature_std:<11.2f} "
        )
    return


def data_partition(cleaned_array):
    """
    A function to split the array into a nested dictionary
    of radii for each density.

    Method
    ------
    This in effectively an implementation of a nested hash map for the radii
    of each density of sphere. We choose this structure to reduce time
    complextiy of accessing specific points (O(1)).

    A single pass of the data is done (O(1)) and new dictionaries are created
    dynamically for unique densities/radii. This method is chosen to allow for
    expandability as it is robust for additional values and does not rely on a
    look-up table or a predefined structure.

    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean

    Returns
    -------
    data_partition : dict
        A nested dictionary with keys:densities (str) and values:dictionaries
        with keys:radii (str) containing lists of data.

    """
    data_partition = {}  # initialise main dictionary
    for row in cleaned_array:  # loop rows
        density = row[0]
        radius = row[1]
        if (
            density not in data_partition
        ):  # create new entry if density not seen before
            data_partition[density] = {}
        if (
            radius not in data_partition[density]
        ):  # create new sub entry if radius not seen before
            data_partition[density][radius] = []
        data_partition[density][radius].append(row)  # add values to dict
    return data_partition


def plotting_drop_vs_fall(data_partition, materials):
    """
    A function to plot drop heigh vs fall time for
    different radii of different materials.

    Method
    ------
    This function generates independent figures for each material to allow for
    better visualisation of how radius impacts fall time, and to prevent
    overcrowding.

    Within each figure (sorted by material (via density)) we use radius as a
    subvariable distinguished by colour. Various numpy functions are used to
    make handling of the data simpler (namely using the [:,x] slicing to
    extract entire columns) and the .astype(float) to convert values to numbers.

    plt.errorbar() is used to maintain integrity, as we know that the time
    values are accurate only to +-1s and the height to +- 10m.

    A simple count variable is used to cycle through materials for the graph
    headers (as they are internally sorted by density).

    Parameters
    ----------
    data_partition : dict
        A nested dictionary with keys:densities (str) and values:dictionaries
        with keys:radii (str) containing lists of data.
    materials : list
        As in data_read_clean.

    Returns
    -------
    None.

    """
    count = 0  # initialise counter for materials
    for (
        density,
        radii,
    ) in data_partition.items():  # loop and select one material
        plt.figure(figsize=(10, 10))  # create figure
        for radius, rows in sorted(
            radii.items(), key=lambda x: float(x[0])
        ):  # loop each radius
            data = np.array(rows)  # generate numpy array
            height = data[:, 5].astype(float)  # select and format data
            time = data[:, 6].astype(float)

            plt.errorbar(
                height,
                time,
                yerr=1,
                xerr=10,
                fmt="o",
                label=f"Radius: {radius}",
                capsize=3,
                elinewidth=1,
            )  # plots points with errors
        plt.title(
            f"Fall time vs Drop height for material = {
                  materials[count]}"
        )
        plt.xlabel("Drop height (m)")
        plt.ylabel("Fall time (s)")
        plt.grid(True)
        plt.legend()
        plt.show()
        count += 1  # to next materials in loop


def plotting_correlation_matrix(cleaned_array, features):
    """
    Plots a colour-coded correlation matrix generated using np.corrcoeff() for
    features against themselves.
    1 is perfect positive correlation, -1 is perfect negative correlation.

    Method
    ------
    np.corrcoerf() is used in this function to generate the Pearson correlation
    coefficients (eq 3) between different features. A grid figure is then
    created and a colourmap (PiYG) assigned. The value of the coefficient
    is then imposed on each grid square to ensure readability of the graph and
    each column labelled.

    This allows us to see not only through numerical values but through a visual
    method (colour) the correlation between features.


    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.
    features : list
        As in data_read_clean.

    Returns
    -------
    None.

    """
    numpy_array = np.array(cleaned_array, dtype=float)  # generates numpy array
    correlation_matrix = np.corrcoef(
        numpy_array, rowvar=False
    )  # generates correlation matrix values
    figure, axes = plt.subplots(figsize=(10, 10))  # generates figure
    image = axes.imshow(
        correlation_matrix, cmap="PiYG", vmin=-1
    )  # assigns colourmap
    plt.colorbar(image)

    row, col = correlation_matrix.shape  # for expandability
    for i in range(row):  # loops each grid-square
        for j in range(col):
            axes.text(
                j, i, f"{correlation_matrix[i, j]:.2f}", ha="center"
            )  # adds correlation value

    labels = [features[i + 1] for i in range(col)]  # labels each row / column

    axes.set_xticks(np.arange(len(labels)))  # arranges labels
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticklabels(labels)  # plots
    axes.set_yticklabels(labels)
    plt.title("Correlation Matrix")
    plt.show()


def linear_regression(cleaned_array, features, units):
    """
    A function to perform linear regression on the cleaned data, to generate
    the beta values in eq(8) of the exercise sheet.
    Method
    ------
    Using the LinearRegression() model from SKLearn we fit our data. We can then
    use the methods .intercept_ and .coef_ to generate our beta_0 and beta_i
    values. These are then printed for further analysis.

    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.
    features : list
        As in data_read_clean.
    units : list
        As in data_read_clean.

    Returns
    -------
    beta_0 : numpy.float64
        Contains the y-intercept value
    beta_i : numpy.ndarray
        Contains the beta values for each feature.
    model : sklearn.linear_model._base.LinearRegression
        The linear regression model function

    """
    np_array = np.array(cleaned_array, dtype=float)  # generates numpy array
    X = np_array[:, :6]  # extracts data
    Y = np_array[:, 6]
    model = LinearRegression()  # generates model
    model.fit(X, Y)  # fits to model

    beta_0 = model.intercept_  # finds beta_0
    beta_i = model.coef_  # finds feature betas
    print("\nLinear Regrssion results")
    print(f"Beta_0 = {beta_0:.3f} s")  # prints beta_0
    for i in range(len(beta_i)):
        print(
            f"{features[i+1]} (Beta_{i+1}) = {beta_i[i]:.3f} s/{units[i+2]}"
        )  # prints each feature beta with units
    return beta_0, beta_i, model


def plotting_fit_using_means(cleaned_array, model, beta_0, beta_i):
    """
    Plots a comparison of experimental data, model predicted data,
    and a linear fit.

    Method
    ------
    We select a target density and radius for the comparison.
    We want to isolate the Height and Time for these target variables.

    1. We filter the data based on the target using numpy slicing.
    2. We loop each radius and let the trained model guess the fall times
    3. We then calculate the mean of each other feature and then generate our
    straight line using these and the beta values of the model.

    We can use this to assess a goodness of fit to see if the linear model
    fits well with the true experimental data.

    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.
    model : sklearn.linear_model.LinearRegression
        Linear regression model
    beta_0 : float
        y-intercept
    beta_i : numpy.ndarray
        array of betas for each feature

    Returns
    -------
    None.

    """
    numpy_array = np.array(cleaned_array, dtype=float)  # generates numpy array

    density = 7874  # target density and radius
    radius = 0.005

    mask_material = numpy_array[
        numpy_array[:, 0] == density
    ]  # mask based on material (density)
    mask_radii = np.unique(mask_material[:, 1])  # mask based on radius
    for radius in mask_radii:  # loop radii

        masked_data = mask_material[mask_material[:, 1] == radius]  # apply mask

        height = masked_data[:, 5]  # extract data from mask
        time = masked_data[:, 6]

        time_predicted = model.predict(
            masked_data[:, :6]
        )  # predict using model

        means = np.mean(masked_data, axis=0)  # calculate mean values
        mean_density, mean_radius, mean_mass, mean_temp, mean_pressure = (
            means[0],
            means[1],
            means[2],
            means[3],
            means[4],
        )

        straight_line = np.array(
            [min(height), max(height)]
        )  # generate a straight line of means

        time_equation = (  # calculate the time values using eq(8)
            beta_0
            + beta_i[0] * mean_density
            + beta_i[1] * mean_radius
            + beta_i[2] * mean_mass
            + beta_i[3] * mean_temp
            + beta_i[4] * mean_pressure
            + beta_i[5] * straight_line
        )

        plt.figure(figsize=(10, 10))
        plt.scatter(
            height, time, color="blue", label="Experimental"
        )  # plot experimental data
        plt.scatter(
            height, time_predicted, color="red", label="Predicted"
        )  # plot predicted model data
        plt.plot(
            straight_line,
            time_equation,
            color="black",
            label="Fit using Means",  # plot means straight line
        )

        plt.xlabel("Height(m)")
        plt.ylabel("Time(s)")
        plt.title(f"Fall time vs Height (3b). Radius = {radius}")
        plt.legend()
        plt.show()


def func_train_test_split(cleaned_array):
    """
    Splits data into 90% training data and 10% test data. Performs regression on
    training data and then tests on test data. Visualises residuals.

    Method
    ------
    We split our data from the numpy array into a set of training data and a set
    of test data using the train_test_split() function from SKLearn. We choose
    90% train and 10% test. A random seed is used to allow for reproducibility
    in testing.

    The R Squared values for both sets is found using the r2_score function.
    If these two values differ by a lot, then we can see that the model
    is not fitting to the actual data.

    The residuals (Y_actual - Y) are calculated and plotted for each radius.
    If the model were perfect these residuals would cluster around the zero
    line.


    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.

    Returns
    -------
    None.

    """
    np_array = np.array(cleaned_array, dtype=float)  # generates numpy array
    X = np_array[:, :6]  # extracts data
    Y = np_array[:, 6]

    X_train, X_test, Y_train, Y_test = train_test_split(  # splits data
        X, Y, test_size=0.1, random_state=67
    )
    model_train = LinearRegression()  # training model
    model_train.fit(X_train, Y_train)

    Y_train_predict = model_train.predict(
        X_train
    )  # predicts using training model
    Y_test_predict = model_train.predict(X_test)

    r2_train = r2_score(Y_train, Y_train_predict)  # generates r2 scores
    r2_test = r2_score(Y_test, Y_test_predict)
    print("\nTraining vs Test Data fit")
    print(f"R2 Training: {r2_train:.4f}, R2 Test: {r2_test:.4f}")

    residual = Y_test - Y_test_predict  # finds residuals

    radii = np.sort(np.unique(X_test[:, 1]))
    residuals = [residual[X_test[:, 1] == radius] for radius in radii]

    plt.figure(figsize=(10, 10))  # plots residuals for each radius
    plt.boxplot(residuals, labels=[f"{radius:.3f}" for radius in radii])
    plt.axhline(0, color="black")
    plt.xlabel("Radius (m)")
    plt.ylabel("Residual")
    plt.title("Residual vs Radius (Test)")
    plt.show()


def sgd_unscaled(cleaned_array):
    """
    Performs Stochastic Gradient Descent method to find beta values.
    This function does NOT scale the data appropriately.

    Method
    ------
    This function is more for visualisation and understanding than any real
    analysis.
    Using a new model (SGDRegressor) we can predict a new set of Y values.
    We can then use the r2_score function to find the R squared between the
    predicted and experimental values. This R squared value will not be in the
    correct range due to the lack of appropriate scaling.

    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.

    Returns
    -------
    r2 : float
        R squared value for this model against experimental data.

    """
    np_array = np.array(cleaned_array, dtype=float)  # generates numpy array
    X = np_array[:, :6]  # extracts data
    Y = np_array[:, 6]

    sgd = SGDRegressor(random_state=67)  # performs SGD method
    sgd.fit(X, Y)

    Y_predict = sgd.predict(X)  # predicts usind SGD

    r2 = r2_score(Y, Y_predict)  # finds R squared value
    print("\nSGD without scaling")
    print(f"R2 unscaled = {r2:.2f}")
    return r2


def sgd_scaled(cleaned_array, features, loss_function):
    """
    Performs Stochastic Gradient Descent method to find beta values.
    This function DOES scale the data appropriately.

    Method
    ------
    This function improves on sgd_unscaled by performing a scaling on the data.
    We use StandardScaler() to scale the data to a Z distribution (0-1) and
    then preform SGD using a given loss function. The R squared score is once
    again generated and our beta values generated. We then perform an inverse
    scaling by generating parameters mean_ and scale_ of our scaler, and
    performing the operations beta_i_unscaled = beta_scaled / scale_
    and beta_0_unscaled = beta_0_scaled - np.sum((beta_scaled * mean_) / scale_)
    to allow us to have meaningful beta values.


    Parameters
    ----------
    cleaned_array : list
        As in data_read_clean.
    features : list
        As in data_read_clean.
    loss_function : string
        The loss function to use.

    Returns
    -------
    r2 : float
        R squared value.
    beta_0_unscaled : float
        y-intercept
    beta_i_unscaled : numpy.ndarray
        beta values for each feature.

    """
    np_array = np.array(cleaned_array, dtype=float)  # generate numpy array
    X = np_array[:, :6]  # extract data
    Y = np_array[:, 6]

    scale = StandardScaler()  # generate scaler (Z)
    X_scaled = scale.fit_transform(X)  # scale data

    sgd_scaled = SGDRegressor(
        loss=loss_function, epsilon=1.35, random_state=67
    )  # perform SGD with given loss function
    sgd_scaled.fit(X_scaled, Y)  # fit model

    r2 = sgd_scaled.score(X_scaled, Y)  # find R squared
    print(f"\nSGD with {loss_function} loss function")
    print(f"R2 scaled = {r2:.4f}")

    mean = scale.mean_  # generate parameters to unscale
    scales = scale.scale_
    beta_scaled = sgd_scaled.coef_  # generated scaled betas
    beta_0_scaled = sgd_scaled.intercept_

    beta_i_unscaled = beta_scaled / scales  # unscale those betas
    beta_0_unscaled = beta_0_scaled - np.sum((beta_scaled * mean) / scales)

    print(f"Back-found Beta_0 = {beta_0_unscaled[0]:.4f}")
    for i in range(len(beta_i_unscaled)):
        print(
            f"Back-found Beta_{i+1} ({features[i+1]}) = {beta_i_unscaled[i]:.4f}"
        )
    return r2, beta_0_unscaled, beta_i_unscaled


def main():
    """
    Main for Exercise.

    Method
    ------
    This is the main loop. We run all the functions in order to generate
    all figures, beta values and r squared values. We use two different loss
    functions in the sgd_scaled value to compare their efficacy.

    Returns
    -------
    None.

    """
    # read and clean
    cleaned_array, materials, features, units = data_read_clean(
        "exercise3data.csv"
    )
    # display data stats
    display_statistics(cleaned_array, materials, features, units)
    # plot drop height vs fall time
    plotting_drop_vs_fall(data_partition(cleaned_array), materials)
    # plot correlation matrix
    plotting_correlation_matrix(cleaned_array, features)
    # perform linear regression
    beta_0, beta_i, model = linear_regression(cleaned_array, features, units)
    # plot comparison graphs
    plotting_fit_using_means(cleaned_array, model, beta_0, beta_i)
    # perform train-test-split analysis
    func_train_test_split(cleaned_array)
    # perform unscaled SGD
    sgd_unscaled(cleaned_array)
    # perform scaled SGD for different loss function
    loss_function = "squared_error"
    sgd_scaled(cleaned_array, features, loss_function)
    loss_function = "huber"
    sgd_scaled(cleaned_array, features, loss_function)


if __name__ == "__main__":
    main()
