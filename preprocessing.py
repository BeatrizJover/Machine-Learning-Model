import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Any


class PropertyDataProcessor:
    """
    A class to preprocess property data for analysis.
    """

    def __init__(self, properties_file: str, codes_file: str) -> None:
        """
        Initialize the processor with file paths.
        Args:
        - properties_file: Path to the property data CSV file.
        - codes_file: Path to the postal codes CSV file.
        """
        self.properties_file = properties_file
        self.codes_file = codes_file
        self.df = None
        self.df_codes = None

    def load_data(self) -> None:
        """
        Load data from the specified CSV files.
        """
        try:
            self.df = pd.read_csv(self.properties_file)
            self.df_codes = pd.read_csv(self.codes_file)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise

    def clean_and_merge_data(self) -> None:
        """
        Clean and merge the property and postal codes data.
        """
        # Drop unnecessary columns from the codes file
        self.df_codes = self.df_codes.drop(
            columns=[
                "NIS-code Region",
                "Region_FR",
                "Region_NL",
                "Region_EN",
                "NIS-code Municipaity",
                "Municipality_FR",
                "Municipality_NL",
            ]
        )
        # Merge datasets on postal codes
        self.df = (
            pd.merge(
                self.df_codes,
                self.df,
                left_on="Postal code",
                right_on="Postal_Code",
                how="outer",
            )
            .drop(columns=["Postal code"])
            .dropna()
        )

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two geographical points.
        Args:
        - lat1, lon1: Latitude and longitude of the first point.
        - lat2, lon2: Latitude and longitude of the second point.
        Returns:
        - Distance in kilometers as a float.
        """
        earth_radius_km = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return round(earth_radius_km * c, 2)

    def calculate_distances(self, city_coords: Dict[str, Dict[str, float]]) -> None:
        """
        Add columns for distances to specified cities.
        Args:
        - city_coords: A dictionary of city names and their coordinates.
        """
        for city, coords in city_coords.items():
            self.df[f"Dist_{city}"] = self.df.apply(
                lambda row: self.haversine_distance(
                    row["Latitude"],
                    row["Longitude"],
                    coords["latitude"],
                    coords["longitude"],
                ),
                axis=1,
            )
        self.df["Dist_nearest_city"] = self.df[
            [f"Dist_{city}" for city in city_coords.keys()]
        ].min(axis=1)

    def transform_features(self) -> None:
        """
        Transform and engineer features for further analysis.
        """
        # Map province to regions
        province_to_region = {
            "Brussels-Capital Region": "Brussels-Capital Region",
            "Antwerp": "Flemish Region (Flanders)",
            "Flemish Brabant": "Flemish Region (Flanders)",
            "West Flanders": "Flemish Region (Flanders)",
            "East Flanders": "Flemish Region (Flanders)",
            "Limburg": "Flemish Region (Flanders)",
            "Walloon Brabant": "Walloon Region (Wallonia)",
            "Hainaut": "Walloon Region (Wallonia)",
            "Liège": "Walloon Region (Wallonia)",
            "Luxembourg": "Walloon Region (Wallonia)",
            "Namur": "Walloon Region (Wallonia)",
        }
        self.df["Region"] = self.df["Province"].map(province_to_region)

        region_numeric_mapping = {
            "Brussels-Capital Region": 1,
            "Flemish Region (Flanders)": 2,
            "Walloon Region (Wallonia)": 3,
        }
        self.df["Region_Numeric"] = self.df["Region"].map(region_numeric_mapping)

        # Replace zero rooms with 1 and calculate combined metrics
        self.df["Number_of_rooms"] = self.df["Number_of_rooms"].replace(0, 1)
        self.df["Room_Space_Combined"] = (
            self.df["Living_Area"] * self.df["Number_of_rooms"]
        )
        self.df["Outside_area"] = self.df[
            ["Terrace_Area", "Garden_Area", "Surface_of_the_land"]
        ].sum(axis=1)

        # Clean and encode kitchen and building state
        self.df["Kitchen"] = self.df["Kitchen"].replace("Does not specify", None)
        self.df["Building_State"] = self.df["Building_State"].replace(
            "Does not specify", None
        )
        self.df["Kitchen"] = np.select(
            [self.df["Kitchen"].isnull(), self.df["Kitchen"] == "NOT_INSTALLED"],
            [1, 0],
            default=2,
        )
        renovation_states = ["TO_RENOVATE", "TO_BE_DONE_UP", "TO_RESTORE"]
        self.df["Building_State"] = np.select(
            [
                self.df["Building_State"].isnull(),
                self.df["Building_State"].isin(renovation_states),
            ],
            [1, 0],
            default=2,
        )
        # Filter out subtypes with at least 20 samples and map subtypes to their respective counts
        subtype_count = self.df["Subtype_of_property"].value_counts()
        valid_subtypes = subtype_count[subtype_count >= 20].index
        self.df = self.df[self.df["Subtype_of_property"].isin(valid_subtypes)]
        self.df["Subtype_category"] = self.df["Subtype_of_property"].map(subtype_count)

    def save_to_csv(self, output_path: str) -> None:
        """
        Save the processed data to a CSV file.
        Args:
        - output_path: Path to save the processed CSV file.
        """
        self.df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
