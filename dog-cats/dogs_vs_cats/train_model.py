from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db = h5py.File(args["dataset"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] tuning hyperpararmeters...")
params = {"C":[0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("INFO] best hyperparameters:{}".format(model.best_params_))

print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["labels_names"]))

acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score:{}".format(acc))

print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.beat_estimator_))
f.close()

db.close()
