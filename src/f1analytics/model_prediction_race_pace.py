import matplotlib.pyplot as plt
import numpy as np
import fastf1.plotting
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#from sklearn.cluster import KMeans
from pathlib import Path

# FastF1's default color scheme
fastf1.plotting.setup_mpl(color_scheme='fastf1')

def build_training_data():
    # List of non-sprint events in the 2024 F1 calendar
    non_sprint_events = [
        'Bahrain Grand Prix',
        'Saudi Arabian Grand Prix',
        'Australian Grand Prix',
        'Japanese Grand Prix',
        'Emilia Romagna Grand Prix',
        'Canadian Grand Prix',
        'Spanish Grand Prix',
        'British Grand Prix',
        'Hungarian Grand Prix',
        'Belgian Grand Prix',
        'Dutch Grand Prix',
        'Italian Grand Prix',
        'Singapore Grand Prix',
        'Mexican Grand Prix',
        'Las Vegas Grand Prix',
        'Abu Dhabi Grand Prix'
    ]

    transformed_laps_dataframes = {}


    for event in non_sprint_events:
        try:
            # Load the FP2 session
            session = fastf1.get_session(2024, event, 'FP2')
            session.load()
            
            # Filter out box laps and keep only accurate laps
            laps = session.laps.pick_wo_box()
            laps = laps[laps['IsAccurate'] == True]
            
            # Transform laps: add a new column for lap times in seconds
            transformed_laps = laps.copy()
            transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()
            
            # Store the transformed dataframe in the dictionary
            transformed_laps_dataframes[event] = transformed_laps
            print(f"Transformed laps for {event} FP2 loaded successfully.")
        except Exception as e:
            print(f"Failed to load data for {event}: {e}")


    def label_race_pace(dataframe, driver, min_time, max_time):
        # Keep LapTime, Stint, LapNumber (and any ID you need like Driver if not implicit)
        driver_data = dataframe.pick_drivers(driver).copy()

        if driver_data.empty:
            print(f"No data for driver: {driver}")
            return None

        # Remove fastest and slowest laps
        n_to_remove = 1
        driver_data = driver_data.sort_values('LapTime (s)').iloc[n_to_remove:-n_to_remove]
        driver_data = driver_data.reset_index(drop=True)

        # ... existing logic to compute LapTimeDifference, Consistency, is_race_pace, etc.
        # Make sure when you create those new columns you do it on driver_data and do not drop 'Stint'.

        # Reset the index after filtering
        driver_data = driver_data.reset_index(drop=True)

        # **2. Assign is_race_pace Based on Range**
        driver_data['is_race_pace'] = 0  # Default to 0
        driver_data.loc[
            (driver_data['LapTime (s)'] >= min_time) & 
            (driver_data['LapTime (s)'] <= max_time), 
            'is_race_pace'
        ] = 1

        # **3. Calculate LapTimeDifference**
        driver_data['LapTimeDifference'] = driver_data['LapTime (s)'].diff(periods = -1 ).abs()

        driver_data = driver_data.dropna(subset=['LapTimeDifference']).reset_index(drop=True)

        # **4. Add Consistency Feature with Non-Overlapping Rolling**
        window_size = 5  # Non-overlapping group size
        max_diff = 3   # Maximum allowed difference for consistency

        # Assign groups for non-overlapping rolling
        driver_data['Group'] = (driver_data.index // window_size)

        # Calculate max, min, and range for each group
        grouped = driver_data.groupby('Group')['LapTime (s)'].agg(['max', 'min']).reset_index()
        grouped['RollingRange'] = grouped['max'] - grouped['min']

        # Map RollingRange back to the original dataframe
        driver_data = driver_data.merge(grouped[['Group', 'RollingRange']], on='Group', how='left')

        # Add a Consistency flag (1 if within range, 0 otherwise)
        driver_data['Consistency'] = (driver_data['RollingRange'] <= max_diff).astype(int)

        # Drop intermediate columns to keep the dataframe clean
        driver_data.drop(columns=['Group', 'RollingRange'], inplace=True)

        return driver_data


    # Bahrain
    bahrain_pia = label_race_pace(transformed_laps_dataframes[non_sprint_events[0]], 'PIA', 95, 98).reset_index()
    bahrain_oco = label_race_pace(transformed_laps_dataframes[non_sprint_events[0]], 'OCO', 97, 100).reset_index()
    bahrain_bot = label_race_pace(transformed_laps_dataframes[non_sprint_events[0]], 'BOT', 97, 99).reset_index()
    bahrain_ver = label_race_pace(transformed_laps_dataframes[non_sprint_events[0]], 'VER', 96, 98).reset_index()

    # Saudi Arabian
    saudi_lec = label_race_pace(transformed_laps_dataframes[non_sprint_events[1]], 'LEC', 93, 96).reset_index()
    saudi_hul = label_race_pace(transformed_laps_dataframes[non_sprint_events[1]], 'HUL', 94.5, 96.5).reset_index()
    saudi_alb = label_race_pace(transformed_laps_dataframes[non_sprint_events[1]], 'ALB', 94, 96).reset_index()


    # Australia
    australia_nor = label_race_pace(transformed_laps_dataframes[non_sprint_events[2]], 'NOR', 82.5, 84).reset_index()
    australia_lec = label_race_pace(transformed_laps_dataframes[non_sprint_events[2]], 'LEC', 82, 84).reset_index()
    australia_str = label_race_pace(transformed_laps_dataframes[non_sprint_events[2]], 'STR', 82, 84).reset_index()
    australia_rus = label_race_pace(transformed_laps_dataframes[non_sprint_events[2]], 'RUS', 83, 85).reset_index()


    # Imola
    imola_mag = label_race_pace(transformed_laps_dataframes[non_sprint_events[4]], 'MAG', 81, 83).reset_index()
    imola_alo = label_race_pace(transformed_laps_dataframes[non_sprint_events[4]], 'ALO', 81, 83).reset_index()
    imola_zho = label_race_pace(transformed_laps_dataframes[non_sprint_events[4]], 'ZHO', 81, 83).reset_index()

    #Spain

    spain_ver = label_race_pace(transformed_laps_dataframes[non_sprint_events[6]], 'VER', 79, 82).reset_index()
    spain_nor = label_race_pace(transformed_laps_dataframes[non_sprint_events[6]], 'NOR', 78, 81.5).reset_index()
    spain_ham = label_race_pace(transformed_laps_dataframes[non_sprint_events[6]], 'HAM', 79, 82).reset_index()

    #Belgium

    belg_lec = label_race_pace(transformed_laps_dataframes[non_sprint_events[9]], 'LEC', 109, 112).reset_index()
    belg_alb = label_race_pace(transformed_laps_dataframes[non_sprint_events[9]], 'ALB', 108, 111).reset_index()
    belg_bot = label_race_pace(transformed_laps_dataframes[non_sprint_events[9]], 'BOT', 109, 112).reset_index()

    # Step 2: Combine all labeled data
    all_training_data = pd.concat([
        bahrain_pia, bahrain_oco, bahrain_bot, bahrain_ver,
        saudi_lec, saudi_hul, saudi_alb,
        australia_nor, australia_lec, australia_str, australia_rus,
        imola_mag, imola_alo, imola_zho,
        spain_ver, spain_nor, spain_ham,
        belg_lec, belg_alb, belg_bot
    ], ignore_index=True)
    return all_training_data

class RacePaceAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def train_model(self, data=None, test_size=0.2, random_state=42):
        if data is None:
            data = build_training_data()

        data = data.copy()

        feature_cols = [
            'LapTime (s)',
            'LapTimeDifference',
            'Consistency',
            'Stint'
        ]
        X = data[feature_cols]
        y = data['is_race_pace']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, data):
        #Preduct using the trained model
        return self.pipeline.predict(data)
    
    def save_model(self, path: str = 'race_pace_pipeline.pkl'):
        """Save the trained pipeline.
        If `path` is relative, save it next to this module file so it can be loaded reliably regardless of CWD.
        """
        base = Path(__file__).resolve().parent
        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = (base / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        print(f"Pipeline saved to {out_path}")


    def load_model(self, path: str = 'race_pace_pipeline.pkl'):
        """Load a previously saved pipeline.
        If `path` is relative, resolve it next to this module file so it works from any working directory.
        """
        base = Path(__file__).resolve().parent
        in_path = Path(path)
        if not in_path.is_absolute():
            in_path = (base / in_path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found at {in_path}")
        self.pipeline = joblib.load(in_path)
        print(f"Pipeline loaded successfully from {in_path}")
               
    @staticmethod
    def add_consistency_feature(dataframe, window_size, max_diff, n_to_remove=2):
        """
        Add consistency feature to the given dataframe.
        """
        dataframe = dataframe.sort_values('LapTime (s)').iloc[n_to_remove:-n_to_remove].copy()
        dataframe = dataframe.sort_values('LapNumber').reset_index(drop=True)
        dataframe['LapTimeDifference'] = dataframe['LapTime (s)'].diff(periods=-1).abs()
        dataframe = dataframe.dropna(subset=['LapTimeDifference']).reset_index(drop=True)
        dataframe['Group'] = dataframe.index // window_size
        grouped = dataframe.groupby('Group')['LapTime (s)'].agg(['max', 'min']).reset_index()
        grouped['RollingRange'] = grouped['max'] - grouped['min']
        dataframe = dataframe.merge(grouped[['Group', 'RollingRange']], on='Group', how='left')
        dataframe['Consistency'] = (dataframe['RollingRange'] <= max_diff).astype(int)
        dataframe.drop(columns=['Group', 'RollingRange'], inplace=True)
        return dataframe

    def get_race_pace_laps(self, event_data, window_size=3, max_diff=3, n_to_remove=2, lap_time_tolerance=2.0):
        """
        Identify race pace laps for all drivers in a given event.
        """
        drivers_event = event_data['Driver'].unique()
        race_pace_laps_all_drivers = []

        for driver in drivers_event:
            driver_data = event_data.pick_drivers(driver).copy()
            driver_data = driver_data.pick_wo_box()
            driver_data = driver_data[driver_data['IsAccurate'] == True]

            if driver_data.empty:
                continue

            driver_data = self.add_consistency_feature(driver_data, window_size, max_diff, n_to_remove)
            unknown_features = driver_data[['LapTime (s)', 'LapTimeDifference', 'Consistency', 'Stint']]

            if unknown_features.empty:
                continue

            driver_data['is_race_pace'] = self.pipeline.predict(unknown_features)

            race_pace_driver = driver_data[driver_data['is_race_pace'] == 1]
            median_lap_time = race_pace_driver['LapTime (s)'].median()
            lower_bound = median_lap_time - lap_time_tolerance
            upper_bound = median_lap_time + lap_time_tolerance

            race_pace_driver = race_pace_driver[
                (race_pace_driver['LapTime (s)'] >= lower_bound) &
                (race_pace_driver['LapTime (s)'] <= upper_bound)
            ]

            race_pace_laps_all_drivers.append(race_pace_driver)

        if race_pace_laps_all_drivers:
            from fastf1.core import Laps

            session = event_data.session
            raw_df = pd.concat(race_pace_laps_all_drivers, ignore_index=True)

            # Build a proper Laps object (wires the session internally)
            race_pace_laps_all_drivers_df = Laps(raw_df, session=session)
            
            
            # Filter rows where 'LapTime (s)' is within ±3 of the median for the DataFrame
            filtered_race_pace_laps_all_drivers = race_pace_laps_all_drivers_df[
                (race_pace_laps_all_drivers_df['LapTime (s)'] <
                 race_pace_laps_all_drivers_df['LapTime (s)'].median() + 3) &
                (race_pace_laps_all_drivers_df['LapTime (s)'] >
                 race_pace_laps_all_drivers_df['LapTime (s)'].median() - 3)
            ]

            return filtered_race_pace_laps_all_drivers
        

def main():
    analyzer = RacePaceAnalyzer()
    analyzer.train_model()
    analyzer.save_model(path='race_pace_pipeline.pkl')

if __name__ == "__main__":
    main()