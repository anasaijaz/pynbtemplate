<template>
  <div id="app">
    <PynbTemplate code_font_size="0.8rem" width="600px" height="360" :gist="true" :json="json"/>
  </div>
</template>

<script>
import PynbTemplate from './components/PynbTemplate.vue'

export default {
  name: 'App',
  components: {
    PynbTemplate
  },
  data(){
    return {
      json: {
        "cells": [
          {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
              "#Import libraries\n",
              "import math\n",
              "import numpy as np\n",
              "import pandas as pd\n",
              "from collections import Counter\n",
              "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor\n",
              "from sklearn.metrics import accuracy_score\n",
              "from sklearn.model_selection import KFold, StratifiedKFold\n",
              "import time\n",
              "from sklearn.cluster import KMeans\n",
              "from matplotlib.pyplot import figure\n",
              "from sklearn.metrics.cluster import normalized_mutual_info_score"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
              "#Import FRUFS and vevestaX and create vevestaX object\n",
              "from FRUFS import FRUFS\n",
              "from vevestaX import vevesta as v\n",
              "V=v.Experiment()"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
              "# Load the data into a dataframe\n",
              "df = pd.read_csv(\"wine.csv\")\n"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
              {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                  "Data shape:  (178, 13) Target Variable shape:  (178,)\n"
                ]
              }
            ],
            "source": [
              "# Split the data into input features and target variable\n",
              "data, Y = df.drop(['ID'], axis=1), df['ID'].values\n",
              "\n",
              "# Lets check out the shape of our data\n",
              "print(\"Data shape: \", data.shape, \"Target Variable shape: \", Y.shape)"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
              {
                "data": {
                  "text/plain": [
                    "Index(['Alcohol', 'Malic acid', ' Ash', 'Alcalinity of ash', 'Magnesium',\n",
                    "       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',\n",
                    "       'Proanthocyanins', 'Color intensity', 'Hue',\n",
                    "       'OD280/OD315 of diluted wines', 'Proline'],\n",
                    "      dtype='object')"
                  ]
                },
                "execution_count": 5,
                "metadata": {},
                "output_type": "execute_result"
              }
            ],
            "source": [
              "#extract the names of the features\n",
              "V.ds = data\n",
              "#print the names of the features\n",
              "V.ds"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
              "\n",
              "#start the tracking scope of the variables\n",
              "V.start()\n",
              "num_classes = 2"
            ]
          },
          {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
              "# Modelling with FRUFS"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [
              {
                "name": "stderr",
                "output_type": "stream",
                "text": [
                  "Progress bar: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:03<00:00,  3.76it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 624.31it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 537.87it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 627.31it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 693.28it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 634.89it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 625.23it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 781.72it/s]\n",
                  "Progress bar:   0%|                                                                                                                                                                 | 0/13 [00:00<?, ?it/s]"
                ]
              },
              {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                  "Score: 0.8888888888888888\n",
                  "Score: 0.9444444444444444\n",
                  "Score: 1.0\n",
                  "Score: 0.8333333333333334\n",
                  "Score: 0.9444444444444444\n",
                  "Score: 1.0\n",
                  "Score: 0.8888888888888888\n",
                  "Score: 0.8333333333333334\n"
                ]
              },
              {
                "name": "stderr",
                "output_type": "stream",
                "text": [
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 508.78it/s]\n",
                  "Progress bar: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 569.97it/s]"
                ]
              },
              {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                  "Score: 0.9411764705882353\n",
                  "Score: 0.7647058823529411\n",
                  "\n",
                  "\n",
                  "Average Accuracy:  0.903921568627451\n",
                  "\n",
                  "\n",
                  "Total Time Required (in seconds):  3.7195825576782227\n"
                ]
              },
              {
                "name": "stderr",
                "output_type": "stream",
                "text": [
                  "\n"
                ]
              }
            ],
            "source": [
              "# We want to time our algorithm\n",
              "start = time.time()\n",
              "\n",
              "NoOfSplits = 10\n",
              "seed= 27\n",
              "shuffleFlag = True\n",
              "\n",
              "# Use KFold for understanding the performance\n",
              "kfold = KFold(n_splits=NoOfSplits, random_state=seed, shuffle=shuffleFlag)\n",
              "\n",
              "# This will hold all the accuracy scores\n",
              "scores = list()\n",
              "\n",
              "# Perform CV\n",
              "for train, test in kfold.split(data):\n",
              "    # Split data into train and test based on folds\n",
              "    x_train, x_test = data.iloc[train], data.iloc[test]\n",
              "    y_train, y_test = Y[train], Y[test]\n",
              "    \n",
              "    # Convert the data into numpy arrays\n",
              "    x_train, x_test = x_train.values, x_test.values\n",
              "    \n",
              "    noOfFeaturesSelected=6\n",
              "    \n",
              "    # Initialize the FRUFS object with your supervised algorithm of choice\n",
              "    model = FRUFS(model_r=DecisionTreeRegressor(random_state=seed), k=noOfFeaturesSelected, n_jobs=-1, verbose=0, random_state=seed)\n",
              " \n",
              "    # Train the FRUFS model and use it to downsize your data\n",
              "    x_train = model.fit_transform(x_train)\n",
              "    x_test = model.transform(x_test)\n",
              "    \n",
              "    # Finally, classify on selected features\n",
              "    model_dt = DecisionTreeClassifier(random_state=seed)\n",
              "    model_dt.fit(x_train, y_train)\n",
              "    preds = model_dt.predict(x_test)\n",
              "\n",
              "    # We are going to use the NMI metric to measure the quality/performance of the clustering \n",
              "    score = accuracy_score(y_test, preds)\n",
              "    print(\"Score:\", score)\n",
              "    scores.append(score)\n",
              "    \n",
              "# Compute average score\n",
              "averageAccuracy = sum(scores)/len(scores)\n",
              "print(\"\\n\\nAverage Accuracy: \", averageAccuracy)\n",
              "\n",
              "# Finally, check out the total time taken\n",
              "end = time.time()\n",
              "timeTaken = end-start\n",
              "print(\"\\n\\nTotal Time Required (in seconds): \", timeTaken)"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [
              {
                "data": {
                  "text/plain": [
                    "{'num_classes': 2,\n",
                    " 'start': 1648982740.435321,\n",
                    " 'NoOfSplits': 10,\n",
                    " 'seed': 27,\n",
                    " 'shuffleFlag': True,\n",
                    " 'noOfFeaturesSelected': 6,\n",
                    " 'score': 0.7647058823529411,\n",
                    " 'averageAccuracy': 0.903921568627451,\n",
                    " 'end': 1648982744.1549037,\n",
                    " 'timeTaken': 3.7195825576782227}"
                  ]
                },
                "execution_count": 8,
                "metadata": {},
                "output_type": "execute_result"
              }
            ],
            "source": [
              "#end the tracking scope of variables\n",
              "V.end()"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [
              {
                "data": {
                  "image/png": "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\n",
                  "text/plain": [
                    "<Figure size 800x2000 with 1 Axes>"
                  ]
                },
                "metadata": {
                  "needs_background": "light"
                },
                "output_type": "display_data"
              }
            ],
            "source": [
              "figure(figsize=(8, 20), dpi=100)\n",
              "model.feature_importance()"
            ]
          },
          {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [
              {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                  "Dumped the experiment in the file vevesta.xlsx\n",
                  "For additional features, explore our tool at https://www.vevesta.com?utm_source=vevestaX for free.\n"
                ]
              }
            ],
            "source": [
              "V.dump(techniqueUsed = \"Decision tree with FRUFS\",message= \"4 selected features were used\", version=1)"
            ]
          }
        ],
        "metadata": {
          "kernelspec": {
            "display_name": "Python [conda env:vevesta] *",
            "language": "python",
            "name": "conda-env-vevesta-py"
          },
          "language_info": {
            "codemirror_mode": {
              "name": "ipython",
              "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.8"
          }
        },
        "nbformat": 4,
        "nbformat_minor": 4
      }
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}
</style>
