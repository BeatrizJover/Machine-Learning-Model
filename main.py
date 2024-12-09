import time
from preprocessing import PropertyDataProcessor
from modeling import RealEstateModel

def main():
    """
    Main function to execute the end-to-end pipeline for data preprocessing and modeling.
    """
    # Paths to data files
    raw_data_path = "data/belgium_properties_data.csv"
    postal_codes_path = "data/codes-ins-nis-postaux-belgique.csv"
    preprocessed_data_path = "data/preprocess_properties_data.csv"

    # Preprocessing
    print("Starting data preprocessing...")
    preprocessor = PropertyDataProcessor(raw_data_path, postal_codes_path)

    # Sequentially execute preprocessing steps
    preprocessor.load_data()
    preprocessor.clean_and_merge_data()

    # Define coordinates for distance calculation
    belgium_coordinates = {
        "Antwerpen": {"latitude": 51.2194485, "longitude": 4.4024644},
        "Leuven": {"latitude": 50.8798194, "longitude": 4.7004614},
        "Brugge": {"latitude": 51.2093453, "longitude": 3.2247013},
        "Gent": {"latitude": 51.0543384, "longitude": 3.7174184},
        "Hasselt": {"latitude": 50.9306783, "longitude": 5.3373844},
        "Wavre": {"latitude": 50.7175865, "longitude": 4.6119332},
        "Mons": {"latitude": 50.4542209, "longitude": 3.9567027},
        "Liège": {"latitude": 50.6292239, "longitude": 5.5796765},
        "Arlon": {"latitude": 49.6833798, "longitude": 5.8166743},
        "Namur": {"latitude": 50.4669009, "longitude": 4.8674819},
        "Brussels": {"latitude": 50.8503438, "longitude": 4.3517103},
        "Charleroi": {"latitude": 50.4113064, "longitude": 4.4447003},
    }
    preprocessor.calculate_distances(belgium_coordinates)
    preprocessor.transform_features()
    preprocessor.save_to_csv(preprocessed_data_path)
    print(
        f"Data preprocessing completed. Saved preprocessed data to {preprocessed_data_path}.\n"
    )

    # Modeling
    print("Starting model training and evaluation...")
    features = [
        "Room_Space_Combined",
        "Outside_area",
        "Postal_Code",
        "Region_Numeric",
        "Kitchen",
        "Swimming_pool",
        "Building_State",
        "Number_of_frontages",
        "Province_numeric",
        "Subtype_category",
        "Dist_Antwerpen",
        "Dist_Leuven",
        "Dist_Brugge",
        "Dist_Gent",
        "Dist_Hasselt",
        "Dist_Wavre",
        "Dist_Mons",
        "Dist_Liège",
        "Dist_Arlon",
        "Dist_Namur",
        "Dist_Brussels",
        "Dist_Charleroi",
    ]
    target = "Price"
    features_outliers_rem = ["Room_Space_Combined", "Outside_area"]

    model = RealEstateModel(preprocessed_data_path, features, target)
    X_train, X_test, y_train, y_test = model.preprocess(features_outliers_rem)

    start_time = time.time()
    random_search = model.train(X_train, y_train)
    metrics_dict = model.evaluate(X_test, y_test, X_train, y_train)
    end_time = time.time()

    # Print performance metrics
    print(
        f"\nModel training and evaluation completed in {round(end_time - start_time, 1)}s.\n"
    )
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

    # Perform SHAP analysis
    print("Performing SHAP analysis...")
    model.shap_analysis(X_train, X_test, features)

    # Plot predictions
    model.plot_predictions(y_test, random_search.best_estimator_.predict(X_test))

if __name__ == "__main__":
    main()
