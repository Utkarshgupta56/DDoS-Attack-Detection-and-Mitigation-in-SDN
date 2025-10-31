from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import ipaddress

# Convert IPv4 address to integer
def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except Exception:
        return 0

class MachineLearning:
    def __init__(self):
        print("Loading dataset ...")
        df = pd.read_csv(
            '/home/som/Desktop/DDOS Project/DDoS-attack-Detection-and-mitigation-in-SDN/FlowStatsfile.csv'
        )
        for col in ['ip_src', 'ip_dst']:
            if col in df.columns:
                df[col] = df[col].apply(ip_to_int)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1)
        self.df = df

    def flow_training(self):
        print("Flow Training ...")
        data = self.df.copy()
        data = data.iloc[:, 1:]
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.5, shuffle=True, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42
        )

        classifier = RandomForestClassifier(
            n_estimators=30,
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='log2',
            random_state=4,
            class_weight='balanced'
        )
        model = classifier.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("------------------------------------------------------------------------------")

        cm = confusion_matrix(y_test, model.predict(X_test))
        print("Confusion Matrix (Test Set)")
        print(cm)
        print(f"Success accuracy = {test_acc*100:.2f} %")
        print(f"Fail accuracy = {(1-test_acc)*100:.2f} %")
        print("------------------------------------------------------------------------------")

        labels = ['TP', 'FP', 'FN', 'TN']
        counts = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.figure(figsize=(6,4))
        plt.title("Random Forest - Confusion Matrix (Test)")
        plt.bar(labels, counts)
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")


def main():
    ml = MachineLearning()
    ml.flow_training()
if __name__ == "__main__":
    main()