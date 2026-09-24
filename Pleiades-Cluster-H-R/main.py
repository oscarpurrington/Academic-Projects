# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:13:53 2026

@author: OscarLocal
"""
#%% 
"""
Imports all necessary libraries
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


#%%
#------------------
# Part 1
#------------------

#Loads Full CSV file.
df_full = pd.read_csv("m45-clean.csv")

#Filters file based on:
#Positive values
#Relative Error of less than 20%
#Not too extreme pmra and pmdec
df_filtered = df_full[
    (df_full["parallax"] > 0) &
    ( (df_full["parallax"] / df_full["parallax_error"]) >= 5) &
     ( df_full["pmra"].abs() < 200) & 
      (df_full["pmdec"].abs() < 200) ] 

#Removes NaN
df_main = df_filtered.dropna()

#Plots the unfiltered Right Ascension vs Declination
plt.figure(0)
plt.scatter(df_full["pmra"], df_full["pmdec"],marker = "*", s = 12)
plt.xlabel("Proper Motion Right Ascension / mas/yr")
plt.ylabel("Proper Motion Declination / mas/yr")
plt.title("pmra vs pmdec (Unfiltered)")

#Plots the filtered Right Ascension vs Declination
plt.figure(1)
plt.scatter(df_filtered["pmra"], df_filtered["pmdec"],marker = "*", s = 1)
plt.xlabel("Proper Motion Right Ascension / mas/yr")
plt.ylabel("Proper Motion Declination / mas/yr")
plt.title("pmra vs pmdec (Filtered)")

#%%
#------------------
# Part 2
#------------------

#Initialise Scaler (zero mean)
scaler = StandardScaler()

#Scale Data
scaled = scaler.fit_transform(df_main)

#Convert back to Pandas
df_scaled = pd.DataFrame(
    scaled, columns=df_main.columns, index=df_main.index)


#%%
#------------------
# Part 3
#------------------

#Isolates pmra and pmdec values
df_cluster = df_scaled[["pmra","pmdec"]].values

#Variable no. for K-Means and Gaussian - to be changed to indicate how the functions work
n = 10

#Performs K-means seperations with n clusters.
k_means = KMeans(n_clusters=n, n_init=10, random_state=67)
k_means_label = k_means.fit_predict(df_cluster)

#Plots K-means groups by colour
plt.figure(2) 
plt.scatter(df_main["pmra"],df_main["pmdec"],c=k_means_label, s=1, cmap="coolwarm")
plt.title("K-means (n={})".format(n))
plt.xlabel("pmra (m\"/yr)")
plt.ylabel("pmdec (m\"/yr)")

#Performs Gaussian Mixture with n components
gaussian_mixture = GaussianMixture(n_components=n, covariance_type="full",random_state=67)
gaussian_mixture_labels = gaussian_mixture.fit_predict(df_cluster)

#Plots Gaussian Mixture groups by colour
plt.figure(3) 
plt.scatter(df_main["pmra"],df_main["pmdec"],c=gaussian_mixture_labels, s=1, cmap="coolwarm")
plt.title("Gaussian Mixture (n={})".format(n))
plt.xlabel("pmra (m\"/yr)")
plt.ylabel("pmdec (m\"/yr)")


#Saves the results back to df_main
df_main["cluster_label"] = gaussian_mixture_labels

#Prints how many results are in each cluster
print("Counts:", df_main["cluster_label"].value_counts())

#Chooses the 'interesting' cluster (value 2 found from which cluster has approx 1000 values)
cluster_interesting = df_main[df_main["cluster_label"] == 2].copy()
print(f"Cluster of interest holds {len(cluster_interesting)} stars. {(len(cluster_interesting)*100/len(df_full)):2f}% of total.")

#%%
#------------------
# Part 4
#------------------

#Selects the features from the interesting cluster
features = cluster_interesting.drop(columns="cluster_label")

#Performs scaling (zero mean)
scaler_cluster = StandardScaler()
df_cluster_scaled = scaler_cluster.fit_transform(features)

#Finds Covariance Matrix to find correlation
C = np.cov(df_cluster_scaled, rowvar=False)

#Finds principle axes of variance
eigenvalues, eigenvectors = np.linalg.eigh(C)

#Flips s.t. most important feature is first
eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:,::-1]

#Finds the percentage of the variance explained by each feature
explained_variance = eigenvalues/np.sum(eigenvalues)

#Prints the Principle Components
print(f"""Principle Component 1 {eigenvectors[:,0]} explains {(explained_variance[0]*100):2f}%
      Principle Component 2 {eigenvectors[:,1]} explains {(explained_variance[1]*100):2f}%""")

#%%
#------------------
# Part 5
#------------------

#Uses matrix multiplication to project the original data onto a 2D space defined by the principle components
principle_component_projection = df_cluster_scaled @ eigenvectors[:,:2]

#Plots the principle component projection
plt.figure(4)
plt.scatter(principle_component_projection[:,0],principle_component_projection[:,1],s=5,)
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.title("Principle Component Analysis Correlation Scatter")

#Calculates which original features are most important for each of the two principle components.
loadings_1 = eigenvectors[:,0] * np.sqrt(eigenvalues[0])
loadings_2 = eigenvectors[:,1] * np.sqrt(eigenvalues[1])

#Creates a dataframe of the features and their loadings.
df_loadings = pd.DataFrame({
    "Feature": features.columns,
    "Loadings_Principle_Component_1":loadings_1,
    "Loadings_Principle_Component_2":loadings_2
})

#Prints the top 3 features for each principle component.
top_3_1 = df_loadings.reindex(df_loadings["Loadings_Principle_Component_1"].abs().nlargest(3).index)
top_3_2 = df_loadings.reindex(df_loadings["Loadings_Principle_Component_2"].abs().nlargest(3).index)
print(f"""Top 3 Features for Principle Component 1: {top_3_1}
      Top 3 Features for Principle Component 2: {top_3_2}""")

#Plots the Hertzsprung-Russell Diagram
plt.figure(5)
colour_index = cluster_interesting["bp"] - cluster_interesting["rp"]
magnitude = cluster_interesting["g"]
plt.scatter(colour_index, magnitude, s=10)
#Key Step: Inverts y-axis to put in the correct format for an HR diagram.
plt.gca().invert_yaxis()
plt.xlabel("Colour Index")
plt.ylabel("Apparent Magnitude")
plt.title("Hertzsprung-Russell Diagram")

plt.show()
