import os
import pandas as pd
import numpy as np
import json
import mplsoccer

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import statsmodels.api as sm
import statsmodels.formula.api as smf

import sklearn as sk

# Create an empty dataframe that will store top 5 league data
top5leaguedata = pd.DataFrame()

# Our list of league filenames
league_files = ['events_England.json', 'events_France.json', 'events_Germany.json', 'events_Italy.json', 'events_Spain.json']

for filename in league_files:
    file_path = os.path.join('C:/Users/Edin/Downloads/events', filename)

    # Read and load the JSON data
    with open(file_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    top5leaguedata = pd.concat([top5leaguedata, df], ignore_index=True)

print(top5leaguedata.head())
print(top5leaguedata.info())

def shot_matrix(eventdata):

    shots = pd.DataFrame(columns=['Goal','x','y','playerid','teamid','matchid','header'])

    all_shots = eventdata[eventdata['subEventName'] == 'Shot']

    for index,shot in all_shots.iterrows():
        shots.at[index,'Goal']=0
        shots.at[index,'header']=0
        for tag  in shot['tags']:
                if tag['id']==101:
                     shots.at[index,'Goal']=1
                elif tag['id']==403:
                     shots.at[index,'header']=1

        shots.at[index, 'Y']=shot['positions'][0]['y']*.68
        shots.at[index, 'X']= (100 - shot['positions'][0]['x'])*1.05

        shots.at[index,'x']= 100 - shot['positions'][0]['x'] 
        shots.at[index,'y']=shot['positions'][0]['y']
        shots.at[index,'Center_dis']=abs(shot['positions'][0]['y']-50)

        x = shots.at[index,'x']*1.05
        y = shots.at[index,'Center_dis']*.68
        shots.at[index,'Distance'] = np.sqrt(x**2 + y**2)

        #we are interested in the angle made between the width of the goal and the
        #straight line distance to the shot location. A goal is 7.32 meters wide
        #use the law of cosines
        c=7.32
        a=np.sqrt((y-7.32/2)**2 + x**2)
        b=np.sqrt((y+7.32/2)**2 + x**2)
        k = (c**2-a**2-b**2)/(-2*a*b)
        gamma = np.arccos(k)
        if gamma<0:
            gamma = np.pi + gamma
        shots.at[index,'Angle Radians'] = gamma
        shots.at[index,'Angle Degrees'] = gamma*180/np.pi
        
        #lastly we add the identifiers for player, team and match
        shots.at[index,'playerid']=shot['playerId']
        shots.at[index,'matchid']=shot['matchId']
        shots.at[index,'teamid']=shot['teamId']


    return shots

shots = shot_matrix(top5leaguedata)

# Print the number of nulls in each column
print(shots.isnull().sum())

print(shots.head(5))
shots.dropna(inplace=True)
shots = shots[shots['Distance'] <= 50]


goals = shots[shots['Goal'] == 1]

from mplsoccer import Pitch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define shots and goals that are headers
shots_header = shots[shots['header'] == 1]
goals_header = goals[goals['header'] == 1]

# Define shots and goals that are non-headers
shots_non_header = shots[shots['header'] == 0]
goals_non_header = goals[goals['header'] == 0]


# Create the pitch
pitch = Pitch(pitch_type='opta')  # plotting an Opta/Stats Perform pitch

# Create the formatter
formatter = FuncFormatter(lambda x, pos: f'{x:.0f}')

# Plot the non-header shots
fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(20, 9))
plt.suptitle("Non-header shots", fontsize=18)

# Hexbins for non-header shots
hb_shots = axs[0].hexbin(shots_non_header['x'], shots_non_header['y'], gridsize=20, cmap='Blues', 
                      mincnt=1, alpha=0.5)
axs[0].set_title('Shot Density')
cb1 = fig.colorbar(hb_shots, ax=axs[0], format=formatter)
cb1.set_label('Shot Frequency')

# Hexbins for non-header goals
hb_goals = axs[1].hexbin(goals_non_header['x'], goals_non_header['y'], gridsize=20, cmap='Reds', 
                      mincnt=1, alpha=0.5)
axs[1].set_title('Goal Density')
cb2 = fig.colorbar(hb_goals, ax=axs[1], format=formatter)
cb2.set_label('Goal Frequency')

plt.show()

# Plot the header shots
fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(20, 9))
plt.suptitle("Header shots", fontsize=18)

# Hexbins for header shots
hb_shots = axs[0].hexbin(shots_header['x'], shots_header['y'], gridsize=20, cmap='Blues', 
                      mincnt=1, alpha=0.5)
axs[0].set_title('Shot Density')
cb1 = fig.colorbar(hb_shots, ax=axs[0], format=formatter)
cb1.set_label('Shot Frequency')

# Hexbins for header goals
hb_goals = axs[1].hexbin(goals_header['x'], goals_header['y'], gridsize=20, cmap='Reds', 
                      mincnt=1, alpha=0.5)
axs[1].set_title('Goal Density')
cb2 = fig.colorbar(hb_goals, ax=axs[1], format=formatter)
cb2.set_label('Goal Frequency')

plt.show()



import numpy as np

# Define a function for creating probability plots
def create_prob_plot(ax, shots, goals, title):
    # Create histograms for shots and goals
    hist_shots, xedges, yedges = np.histogram2d(shots['x'], shots['y'], bins=[30, 30], range=[[0, 100], [0, 100]])
    hist_goals, _, _ = np.histogram2d(goals['x'], goals['y'], bins=[30, 30], range=[[0, 100], [0, 100]])

    # Calculate the probability
    prob_goals = hist_goals / hist_shots

    # Replace NaN and inf values (these appear when you divide by zero)
    prob_goals = np.nan_to_num(prob_goals, nan=0.0, posinf=0.0, neginf=0.0)

    # Plot probability density
    pcm = ax.pcolormesh(xedges, yedges, prob_goals.T, cmap='hot', rasterized=True, alpha=0.5)

    # Create a colorbar with custom ticks
    cb = fig.colorbar(pcm, ax=ax, extend='max')
    cb.set_label('Probability')

    # Set ticks
    ticks = np.arange(0, 1.2, 0.2)  # Create ticks from 0 to 1 with step 0.2
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Set tick labels

    ax.set_title(title)

# Create the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')

# Create the figure and axes
fig, axs = pitch.draw(nrows=2, ncols=2, figsize=(16, 16))

# Create probability plots for non-headers and headers
create_prob_plot(axs[0, 0], shots_non_header, goals_non_header, 'Probability Density Function (Non-Headers)')
create_prob_plot(axs[0, 1], shots_header, goals_header, 'Probability Density Function (Headers)')

plt.tight_layout()
plt.show()

def create_1d_prob_plot(ax, shots, goals, variable, title):
    if variable == 'Distance':
        bins = 30
        range_hist = [0, 50]  # limit the range to 50 for distance
    else:
        bins = 30
        range_hist = [0, shots[variable].max()]
        
    shots_hist = np.histogram(shots[variable], bins=bins, range=range_hist)
    goals_hist = np.histogram(goals[variable], bins=bins, range=range_hist)
    prob_goals = np.nan_to_num(goals_hist[0] / shots_hist[0])
    ax.plot(shots_hist[1][:-1], prob_goals, marker='o')
    ax.set_xlabel(variable)
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])  # set y-limit to [0,1]
    ax.set_title(title)


# Create the figure and axes
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

# Create 1D probability plots
create_1d_prob_plot(axs[0, 0], shots_non_header, goals_non_header, 'Distance', 'Probability vs Distance (Non-Headers)')
create_1d_prob_plot(axs[0, 1], shots_non_header, goals_non_header, 'Angle Degrees', 'Probability vs Angle (Non-Headers)')
create_1d_prob_plot(axs[1, 0], shots_header, goals_header, 'Distance', 'Probability vs Distance (Headers)')
create_1d_prob_plot(axs[1, 1], shots_header, goals_header, 'Angle Degrees', 'Probability vs Angle (Headers)')

plt.tight_layout()
plt.show()

shots['log_distance'] = np.log(shots['Distance'])

# Visualizing the distribution of distance before and after log transformation
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.hist(shots['Distance'], bins=30, edgecolor='black')
plt.title('Histogram of Distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.hist(shots['log_distance'], bins=30, edgecolor='black')
plt.title('Histogram of Log Transformed Distance')
plt.xlabel('Log Distance')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Convert 'Goal' column to integer type
shots['Goal'] = shots['Goal'].astype(int)

# Creating log transformed Distance feature
shots['log_distance'] = np.log(shots['Distance'])

# Interaction term for Angle and Distance
shots['angle_distance_interaction'] = shots['Angle Radians'] * shots['Distance']

# List of features to consider in the model
features = ['Distance', 'Angle Radians', 'log_distance', 'angle_distance_interaction']
target = 'Goal'

# Splitting the dataset into training set and test set
X = shots[features]
y = shots[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model on the original data
logit_model_raw = sm.Logit(y_train, sm.add_constant(X_train[['Distance', 'Angle Radians']]))
result_raw = logit_model_raw.fit()

# Fit a logistic regression model on the log-transformed data and interaction term
logit_model_log = sm.Logit(y_train, sm.add_constant(X_train[['log_distance', 'Angle Radians', 'angle_distance_interaction']]))
result_log = logit_model_log.fit()

# Make predictions with both models
y_pred_raw = result_raw.predict(sm.add_constant(X_test[['Distance', 'Angle Radians']]))
y_pred_log = result_log.predict(sm.add_constant(X_test[['log_distance', 'Angle Radians', 'angle_distance_interaction']]))

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of raw data against predictions
ax1.scatter(X_test['Distance'], y_pred_raw, alpha=0.1)
ax1.set_title('Raw Data vs Predictions')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Predicted Probability')

# Scatter plot of log-transformed data against predictions
ax2.scatter(X_test['log_distance'], y_pred_log, alpha=0.1)
ax2.set_title('Log-transformed Data vs Predictions')
ax2.set_xlabel('Log Distance')
ax2.set_ylabel('Predicted Probability')

plt.tight_layout()
plt.show()

# Logistic regression model for Distance
logit_model_dist = smf.glm(formula="Goal ~ Distance",
                           data=shots, family=sm.families.Binomial()).fit()

# Logistic regression model for log-transformed Distance
logit_model_log_dist = smf.glm(formula="Goal ~ np.log(Distance)",
                               data=shots, family=sm.families.Binomial()).fit()

# Create a new dataframe for predictions
pred_data = pd.DataFrame()
pred_data["Distance"] = np.linspace(shots["Distance"].min(), shots["Distance"].max(), num=500)
pred_data["log_Distance"] = np.log(pred_data["Distance"])

# Get predictions
pred_data["Goal_Prob_Dist"] = logit_model_dist.predict(pred_data)
pred_data["Goal_Prob_Log_Dist"] = logit_model_log_dist.predict(pred_data)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Distance vs Predicted Probabilities
axs[0].scatter(shots["Distance"], shots["Goal"], label="Raw Data")
axs[0].plot(pred_data["Distance"], pred_data["Goal_Prob_Dist"], color='red', label="Logistic Model")
axs[0].set_xlabel("Distance")
axs[0].set_ylabel("Goal")
axs[0].legend()

# Log-transformed Distance vs Predicted Probabilities
axs[1].scatter(shots["Distance"], shots["Goal"], label="Raw Data")
axs[1].plot(pred_data["log_Distance"], pred_data["Goal_Prob_Log_Dist"], color='red', label="Logistic Model")
axs[1].set_xlabel("Log Distance")
axs[1].set_ylabel("Goal")
axs[1].legend()

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Create 100 random samples
samples = shots.sample(n=100)

# Bin the data for Distance
samples['dist_bins'] = pd.cut(samples['Distance'], bins=np.linspace(samples["Distance"].min(), samples["Distance"].max(), num=20), include_lowest=True)
goal_ratio_per_dist_bin = samples.groupby('dist_bins')['Goal'].mean()

# Bin the data for log-transformed Distance
samples['log_dist_bins'] = pd.cut(samples['log_distance'], bins=np.linspace(samples["log_distance"].min(), samples["log_distance"].max(), num=20), include_lowest=True)
goal_ratio_per_log_dist_bin = samples.groupby('log_dist_bins')['Goal'].mean()

# Create logistic regression models
logit_model_dist = smf.glm(formula="Goal ~ Distance", data=samples, family=sm.families.Binomial()).fit()
logit_model_log_dist = smf.glm(formula="Goal ~ log_distance", data=samples, family=sm.families.Binomial()).fit()

# Create a new dataframe for predictions
pred_data = pd.DataFrame()
pred_data["Distance"] = np.linspace(samples["Distance"].min(), samples["Distance"].max(), num=500)
pred_data["log_distance"] = np.log(pred_data["Distance"])

# Get predictions
pred_data["Goal_Prob_Dist"] = logit_model_dist.predict(pred_data)
pred_data["Goal_Prob_Log_Dist"] = logit_model_log_dist.predict(pred_data)

# Get midpoints of the bins for the scatter plot
dist_bin_mids = [bin.mid for bin in goal_ratio_per_dist_bin.index.categories]
log_dist_bin_mids = [bin.mid for bin in goal_ratio_per_log_dist_bin.index.categories]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# Distance vs Predicted Probabilities
axs[0].scatter(dist_bin_mids, goal_ratio_per_dist_bin, label="Binned Data")
axs[0].plot(pred_data["Distance"], pred_data["Goal_Prob_Dist"], color='red', label="Logistic Model")
axs[0].set_xlabel("Distance")
axs[0].set_ylabel("Goal Ratio")
axs[0].legend()

# Log-transformed Distance vs Predicted Probabilities
axs[1].scatter(log_dist_bin_mids, goal_ratio_per_log_dist_bin, label="Binned Data")
axs[1].plot(pred_data["log_distance"], pred_data["Goal_Prob_Log_Dist"], color='red', label="Logistic Model")
axs[1].set_xlabel("Log Distance")
axs[1].set_ylabel("Goal Ratio")
axs[1].legend()

plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Define interaction terms
shots['interaction_term'] = shots['Distance'] * shots['Angle Radians']
shots['interaction_term_log'] = np.log(shots['Distance']) * shots['Angle Radians']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(shots[['Distance', 'Angle Radians', 'interaction_term']], 
                                                    shots['Goal'], test_size=0.2, random_state=42)

# Model 1
model1 = LogisticRegression()
model1.fit(X_train, y_train)

# Predict on test set
y_pred1 = model1.predict(X_test)
y_pred_proba1 = model1.predict_proba(X_test)[:, 1]

# Evaluate Model 1
print(f"Model 1 Accuracy: {accuracy_score(y_test, y_pred1)}")
print(f"Model 1 Log Loss: {log_loss(y_test, y_pred_proba1)}")
print(f"Model 1 AUC: {roc_auc_score(y_test, y_pred_proba1)}")

# Model 2
# First, we need to apply the log transformation to distance
X_train_log = X_train.copy()
X_test_log = X_test.copy()
X_train_log['Distance'] = np.log(X_train_log['Distance'])
X_test_log['Distance'] = np.log(X_test_log['Distance'])
X_train_log['interaction_term'] = X_train_log['Distance'] * X_train_log['Angle Radians']
X_test_log['interaction_term'] = X_test_log['Distance'] * X_test_log['Angle Radians']

model2 = LogisticRegression()
model2.fit(X_train_log, y_train)

# Predict on test set
y_pred2 = model2.predict(X_test_log)
y_pred_proba2 = model2.predict_proba(X_test_log)[:, 1]

# Evaluate Model 2
print(f"Model 2 Accuracy: {accuracy_score(y_test, y_pred2)}")
print(f"Model 2 Log Loss: {log_loss(y_test, y_pred_proba2)}")
print(f"Model 2 AUC: {roc_auc_score(y_test, y_pred_proba2)}")

import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Randomly sample 50 shots
sample_shots = shots.sample(n=50, random_state=42)

# Get model predictions
sample_shots["prediction_proba"] = result_log.predict(sm.add_constant(sample_shots[['log_distance', 'Angle Radians', 'angle_distance_interaction']]))

# Convert predictions to binary using 0.5 as a threshold
sample_shots["prediction"] = (sample_shots["prediction_proba"] > 0.5).astype(int)

# Create a column for correct predictions
sample_shots["correct"] = (sample_shots["prediction"] == sample_shots["Goal"]).astype(int)

# Create the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(10, 6))

# Plot the shots
scatter = pitch.scatter(sample_shots["X"], sample_shots["Y"], ax=ax, c=sample_shots["correct"], cmap="coolwarm")

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prediction Correctness')

plt.show()


from sklearn.metrics import roc_auc_score, accuracy_score

# Thresholds to test
thresholds = [0.1, 0.2, 0.3, 0.4]

# For each threshold, calculate and print the AUC and accuracy
for threshold in thresholds:
    print(f"\nThreshold: {threshold}")
    
    for model_name, y_pred in zip(["Raw", "Log"], [y_pred_raw, y_pred_log]):
        # Apply the threshold to the predictions
        y_pred_threshold = (y_pred > threshold).astype(int)
        
        # Calculate the AUC
        auc = roc_auc_score(y_test, y_pred)
        print(f"{model_name} AUC: {auc}")
        
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred_threshold)
        print(f"{model_name} Accuracy: {accuracy}")

#Next steps: 1. Create seperate models for headers and normal shots, and compare their performance
#to the model that has all of them.
#Make an ROC graph that compares all of the models so that we can understand how exactly they are performing.

# For header shots
# Add 'log_distance' and 'angle_distance_interaction' to 'shots_header' and 'shots_non_header' dataframes
# Add 'log_distance' and 'angle_distance_interaction' to 'shots_header' and 'shots_non_header' dataframes
shots_header = shots_header.copy()
shots_header['log_distance'] = np.log(shots_header['Distance'])
shots_header['angle_distance_interaction'] = shots_header['Angle Radians'] * shots_header['Distance']
shots_header['Goal'] = shots_header['Goal'].astype(int)

shots_non_header = shots_non_header.copy()
shots_non_header['log_distance'] = np.log(shots_non_header['Distance'])
shots_non_header['angle_distance_interaction'] = shots_non_header['Angle Radians'] * shots_non_header['Distance']
shots_non_header['Goal'] = shots_non_header['Goal'].astype(int)

# For header shots
X_train_header, X_test_header, y_train_header, y_test_header = train_test_split(
    shots_header[['log_distance', 'Angle Radians', 'angle_distance_interaction']], 
    shots_header['Goal'], 
    test_size=0.2, 
    random_state=42
)

# Model for header shots
model_header = LogisticRegression()
model_header.fit(X_train_header, y_train_header)

# Predict on test set
y_pred_header = model_header.predict(X_test_header)
y_pred_proba_header = model_header.predict_proba(X_test_header)[:, 1]

# Evaluate the model
print(f"Header shots model Accuracy: {accuracy_score(y_test_header, y_pred_header)}")
print(f"Header shots model Log Loss: {log_loss(y_test_header, y_pred_proba_header)}")
print(f"Header shots model AUC: {roc_auc_score(y_test_header, y_pred_proba_header)}")

# For non-header shots
X_train_non_header, X_test_non_header, y_train_non_header, y_test_non_header = train_test_split(
    shots_non_header[['log_distance', 'Angle Radians', 'angle_distance_interaction']], 
    shots_non_header['Goal'], 
    test_size=0.2, 
    random_state=42
)

# Model for non-header shots
model_non_header = LogisticRegression()
model_non_header.fit(X_train_non_header, y_train_non_header)

# Predict on test set
y_pred_non_header = model_non_header.predict(X_test_non_header)
y_pred_proba_non_header = model_non_header.predict_proba(X_test_non_header)[:, 1]

# Evaluate the model
print(f"Non-header shots model Accuracy: {accuracy_score(y_test_non_header, y_pred_non_header)}")
print(f"Non-header shots model Log Loss: {log_loss(y_test_non_header, y_pred_proba_non_header)}")
print(f"Non-header shots model AUC: {roc_auc_score(y_test_non_header, y_pred_proba_non_header)}")

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Convert 'header' column to integers
shots['header'] = shots['header'].astype(int)

# Add new features
shots['distance_angle_interaction'] = shots['Distance'] * shots['Angle Radians']

# Choose features
features = ['Angle Radians', 'Distance', 'header', 'distance_angle_interaction']

# Split data into features and target
X = shots[features]
y = shots['Goal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred_new = model.predict(X_test)
y_pred_proba_new = model.predict_proba(X_test)[:, 1]

import statsmodels.api as sm

# Fit the logistic regression model with statsmodels
logit_model = sm.Logit(y_train, sm.add_constant(X_train))
result = logit_model.fit()

# Print the summary of the model
print(result.summary())

# Compute evaluation metrics for the model
accuracy = accuracy_score(y_test, y_pred_new)
log_loss_value = log_loss(y_test, y_pred_proba_new)
auc = roc_auc_score(y_test, y_pred_proba_new)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {log_loss_value}")
print(f"AUC: {auc}")

# Perform cross-validation and calculate scores
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("Cross-validated Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation of Accuracy:", scores.std())

from sklearn.metrics import f1_score

# Calculate F1 score for the new model
f1 = f1_score(y_test, np.round(y_pred_new))

print(f"New model F1 score: {f1}")

from sklearn.metrics import roc_curve

# Compute ROC curve for each model
fpr_new, tpr_new, _ = roc_curve(y_test, y_pred_proba_new)
fpr_total, tpr_total, _ = roc_curve(y_test, y_pred_log)
fpr_header, tpr_header, _ = roc_curve(y_test_header, y_pred_proba_header)
fpr_non_header, tpr_non_header, _ = roc_curve(y_test_non_header, y_pred_proba_non_header)

# Plot ROC curve for each model
plt.figure(figsize=(10, 7))
plt.plot(fpr_new, tpr_new, label='New shots (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba_new))
plt.plot(fpr_total, tpr_total, label='Total shots (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_log))
plt.plot(fpr_header, tpr_header, label='Header shots (AUC = %0.2f)' % roc_auc_score(y_test_header, y_pred_proba_header))
plt.plot(fpr_non_header, tpr_non_header, label='Non-header shots (AUC = %0.2f)' % roc_auc_score(y_test_non_header, y_pred_proba_non_header))
plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate McFadden's R-squared
y_pred_train_proba = model.predict_proba(X_train)[:, 1]
null_log_likelihood = log_loss(y_train, np.ones_like(y_train) * np.mean(y_train))
model_log_likelihood = log_loss(y_train, y_pred_train_proba)
mcfadden_r2 = 1 - (model_log_likelihood / null_log_likelihood)

print("McFadden's R-squared:", mcfadden_r2)

# ... (previous code)

# ... (previous code)

# ... (previous code)

# Custom data
custom_data = {'Distance': [11], 'Angle Radians': [0.64], 'header': [0], 'distance_angle_interaction': [3.75]}

# Preprocess custom data
custom_features = pd.DataFrame(custom_data, columns=X.columns)  # Ensure column order matches model training data

# Make predictions for custom data
custom_pred_proba = model.predict_proba(custom_features)[:, 1]

# Print predicted probabilities
print("Predicted probabilities for custom data:")
print(custom_pred_proba)

# ... (remaining code)
import numpy as np
import pandas as pd

# Randomly sample 50 points from the testing dataset
random_indices = np.random.choice(X_test.index, size=50, replace=False)
random_samples = X_test.loc[random_indices]

# Use the model to predict goals or misses
predictions = model.predict(random_samples)
probabilities = model.predict_proba(random_samples)[:, 1]

# Get the actual labels for the randomly generated shots
actual_labels = y_test.loc[random_indices]

# Create a DataFrame to display the results
results = pd.DataFrame(index=random_indices)
results['Angle Radians'] = random_samples['Angle Radians']
results['Distance'] = random_samples['Distance']
results['Header'] = random_samples['header']
results['Predicted Probability'] = probabilities
results['Actual Label'] = actual_labels
results['Prediction'] = predictions

# Print the top 10 randomly generated shots
top_10_shots = results.head(10)
print("Top 10 Randomly Generated Shots:")
print(top_10_shots)

# Rest of the code remains the same...

import numpy as np
from matplotlib import colors

# Define a function for creating probability plots using the model
def create_model_prob_plot(ax, model, header, title):
    # Define a grid
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 100, 30)
    xv, yv = np.meshgrid(x, y)

    # Prepare the grid for DataFrame conversion
    grid = np.column_stack((xv.flatten(), yv.flatten()))

    # Create a DataFrame from the grid
    grid_cells = pd.DataFrame(grid, columns=['X', 'Y'])

    # Compute features for the grid
    grid_cells['Angle Radians'] = np.arctan2(np.abs(grid_cells['Y']-50), 100-grid_cells['X'])
    grid_cells['Distance'] = np.sqrt(np.power(grid_cells['Y']-50, 2) + np.power(100-grid_cells['X'], 2))
    grid_cells['header'] = header
    grid_cells['distance_angle_interaction'] = grid_cells['Distance'] * grid_cells['Angle Radians']

    # Predict probabilities using the model
    grid_cells['Predicted Probability'] = model.predict_proba(grid_cells[['Angle Radians', 'Distance', 'header', 'distance_angle_interaction']])[:, 1]

    # Convert the probabilities to a 2D array
    prob_goals = grid_cells['Predicted Probability'].values.reshape(len(y), len(x))

    # Plot probability density
    pcm = ax.pcolormesh(x, y, prob_goals, cmap='hot', rasterized=True, alpha=0.5, norm=colors.Normalize(0.,1.))

    # Create a colorbar with custom ticks
    cb = fig.colorbar(pcm, ax=ax, extend='max')
    cb.set_label('Probability')

    # Set ticks
    ticks = np.arange(0, 1.2, 0.2)  # Create ticks from 0 to 1 with step 0.2
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Set tick labels

    ax.set_title(title)

# Create the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')

# Create the figure and axes
fig, axs = pitch.draw(nrows=2, ncols=1, figsize=(16, 16))

# Create probability plots for non-headers and headers using the models
create_model_prob_plot(axs[0], model, 0, 'Model Probability Density Function (Non-Headers)')
create_model_prob_plot(axs[1], model, 1, 'Model Probability Density Function (Headers)')

plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch

# Convert 'header' column to integers
shots['header'] = shots['header'].astype(int)

# Choose features
features = ['Angle Radians', 'Distance', 'header']

# Split data into features and target
X = shots[features]
y = shots['Goal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred_new = model.predict(X_test)
y_pred_proba_new = model.predict_proba(X_test)[:, 1]

# Fit the logistic regression model with statsmodels
logit_model = sm.Logit(y_train, sm.add_constant(X_train))
result = logit_model.fit()

# Print the summary of the model
print(result.summary())

print(f"No interaction term model accuracy: {accuracy_score(y_test, y_pred_new)}")
print(f"No interaction term model Log Loss: {log_loss(y_test, y_pred_proba_new)}")
print(f"No interaction term model AUC: {roc_auc_score(y_test, y_pred_proba_new)}")

def create_model_prob_plot(ax, model, header, title):
    # Define a grid
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 100, 30)
    xv, yv = np.meshgrid(x, y)

    # Prepare the grid for DataFrame conversion
    grid = np.column_stack((xv.flatten(), yv.flatten()))

    # Create a DataFrame from the grid
    grid_cells = pd.DataFrame(grid, columns=['X', 'Y'])

    # Compute features for the grid
    grid_cells['Angle Radians'] = np.arctan2(np.abs(grid_cells['Y']-50), 100-grid_cells['X'])
    grid_cells['Distance'] = np.sqrt(np.power(grid_cells['Y']-50, 2) + np.power(100-grid_cells['X'], 2))
    grid_cells['header'] = header

    # Predict probabilities using the model
    grid_cells['Predicted Probability'] = model.predict_proba(grid_cells[['Angle Radians', 'Distance', 'header']])[:, 1]

    # Convert the probabilities to a 2D array
    prob_goals = grid_cells['Predicted Probability'].values.reshape(len(y), len(x))

    # Plot probability density
    pcm = ax.pcolormesh(x, y, prob_goals, cmap='hot', rasterized=True, alpha=0.5, norm=colors.Normalize(0.,1.))

    # Create a colorbar with custom ticks
    cb = fig.colorbar(pcm, ax=ax, extend='max')
    cb.set_label('Probability')

    # Set ticks
    ticks = np.arange(0, 1.2, 0.2)  # Create ticks from 0 to 1 with step 0.2
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}' for tick in ticks])  # Set tick labels

    ax.set_title(title)

# Create the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')

# Create the figure and axes
fig, axs = pitch.draw(nrows=2, ncols=1, figsize=(16, 16))

# Create probability plots for non-headers and headers using the models
create_model_prob_plot(axs[0], model, 0, 'Model Probability Density Function (Non-Headers)')
create_model_prob_plot(axs[1], model, 1, 'Model Probability Density Function (Headers)')

plt.tight_layout()
plt.show()

# Compute ROC curve for each model
fpr_new, tpr_new, _ = roc_curve(y_test, y_pred_proba_new)
fpr_total, tpr_total, _ = roc_curve(y_test, y_pred_log)
fpr_header, tpr_header, _ = roc_curve(y_test_header, y_pred_proba_header)
fpr_non_header, tpr_non_header, _ = roc_curve(y_test_non_header, y_pred_proba_non_header)

# Plot ROC curve for each model
plt.figure(figsize=(10, 7))
plt.plot(fpr_new, tpr_new, label='New shots (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba_new))
plt.plot(fpr_total, tpr_total, label='Total shots (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_log))
plt.plot(fpr_header, tpr_header, label='Header shots (AUC = %0.2f)' % roc_auc_score(y_test_header, y_pred_proba_header))
plt.plot(fpr_non_header, tpr_non_header, label='Non-header shots (AUC = %0.2f)' % roc_auc_score(y_test_non_header, y_pred_proba_non_header))
plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
























