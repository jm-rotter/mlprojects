import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def conv_month(m):
        if m == "Jan":
            return 0
        elif m == "Feb":
            return 1
        elif m == "Mar":
            return 2
        elif m == "Apr":
            return 3
        elif m == "May":
            return 4
        elif m == "June":
            return 5
        elif m == "Jul":
            return 6
        elif m == "Aug":
            return 7
        elif m == "Sep":
            return 8
        elif m == "Oct":
            return 9
        elif m == "Nov":
            return 10
        elif m == "Dec":
            return 11
        else:
            print("ERROR in Month parsing")

    with open(filename) as file:
       x = csv.DictReader(file)  
       evidences = []
       labels = []

       for line in x:
           evidence = []
           evidence.append(int(line[x.fieldnames[0]]))
           evidence.append(float(line[x.fieldnames[1]]))
           evidence.append(int(line[x.fieldnames[2]]))
           evidence.append(float(line[x.fieldnames[3]]))
           evidence.append(int(line[x.fieldnames[4]]))
           evidence.append(float(line[x.fieldnames[5]]))
           evidence.append(float(line[x.fieldnames[6]]))
           evidence.append(float(line[x.fieldnames[7]]))
           evidence.append(float(line[x.fieldnames[8]]))
           evidence.append(float(line[x.fieldnames[9]]))
           evidence.append(conv_month(line[x.fieldnames[10]]))
           evidence.append(int(line[x.fieldnames[11]]))
           evidence.append(int(line[x.fieldnames[12]]))
           evidence.append(int(line[x.fieldnames[13]]))
           evidence.append(int(line[x.fieldnames[14]]))
           evidence.append(1 if line[x.fieldnames[15]]=="Returning_Visitor" else 0)
           evidence.append(1 if line[x.fieldnames[16]]=="TRUE" else 0)

           evidences.append(evidence)
           labels.append(1 if line[x.fieldnames[17]]=="TRUE" else 0)

       return (evidences, labels)
            


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)
    return classifier


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sens_divisor = 0
    sens_score = 0
    spec_divisor = 0
    spec_score = 0

    for l,p in zip(labels,predictions):
        if l == 1:
            sens_divisor += 1
            if p == 1:
                sens_score +=1
        if l == 0:
            spec_divisor +=1
            if p == 0:
                spec_score +=1

    if sens_divisor == 0 or spec_divisor == 0:
        return (0,0)


    return (sens_score/sens_divisor, spec_score/spec_divisor)



if __name__ == "__main__":
    main()
