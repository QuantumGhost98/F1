import matplotlib.pyplot as plt
import numpy as np
import fastf1.plotting
import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_recall_fscore_support
)
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from f1analytics.config import REPO_ROOT, logger
from typing import Optional, Dict, Any, Tuple
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
            logger.info(f"Transformed laps for {event} FP2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load data for {event}: {e}")


    def label_race_pace(dataframe, driver, min_time, max_time):
        # Keep LapTime, Stint, LapNumber (and any ID you need like Driver if not implicit)
        driver_data = dataframe.pick_drivers(driver).copy()

        if driver_data.empty:
            logger.warning(f"No data for driver: {driver}")
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
    """ML pipeline for race pace prediction using Random Forest."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.feature_names_ = ['LapTime (s)', 'LapTimeDifference', 'Consistency', 'Stint']

    def train_model(
        self, 
        data: Optional[pd.DataFrame] = None, 
        test_size: float = 0.2, 
        cv: int = 5
    ) -> Dict[str, Any]:
        """Train the model with stratified K-fold cross-validation.
        
        Args:
            data: Training data DataFrame. If None, builds from build_training_data().
            test_size: Fraction of data to hold out for final test evaluation.
            cv: Number of cross-validation folds.
            
        Returns:
            Dictionary containing accuracy, classification report, CV scores, and feature importances.
        """
        if data is None:
            logger.info("Building training data...")
            data = build_training_data()

        data = data.copy()

        feature_cols = self.feature_names_
        X = data[feature_cols]
        y = data['is_race_pace']

        # Train/test split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Cross-validation on training set
        logger.info(f"Performing {cv}-fold stratified cross-validation...")
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_splitter, scoring='accuracy'
        )
        cv_precision = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_splitter, scoring='precision'
        )
        cv_recall = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_splitter, scoring='recall'
        )
        cv_f1 = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_splitter, scoring='f1'
        )
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"CV Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        logger.info(f"CV Recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        logger.info(f"CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # Fit on full training set
        logger.info("Fitting model on full training set...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Get feature importances
        feature_importances = self.get_feature_importance()
        
        results = {
            'cv_accuracy': cv_scores,
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'cv_f1': cv_f1,
            'test_accuracy': test_accuracy,
            'test_classification_report': test_report,
            'feature_importances': feature_importances
        }
        
        return results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model."""
        return self.pipeline.predict(data)
    
    def tune_hyperparameters(
        self,
        data: Optional[pd.DataFrame] = None,
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5,
        scoring: str = 'f1',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning.
        
        Args:
            data: Training data. If None, builds from build_training_data().
            param_grid: Parameter grid for GridSearchCV. If None, uses default grid.
            cv: Number of cross-validation folds.
            scoring: Scoring metric for optimization.
            n_jobs: Number of parallel jobs (-1 for all cores).
            
        Returns:
            Dictionary with best_params, best_score, and cv_results.
        """
        if data is None:
            logger.info("Building training data...")
            data = build_training_data()
            
        data = data.copy()
        
        if param_grid is None:
            param_grid = {
                'classifier__n_estimators': [50, 100, 200, 300],
                'classifier__max_depth': [None, 5, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
            }
        
        feature_cols = self.feature_names_
        X = data[feature_cols]
        y = data['is_race_pace']
        
        logger.info(f"Starting GridSearchCV with {cv}-fold CV and scoring={scoring}...")
        logger.info(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        
        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best {scoring} score: {self.best_score_:.4f}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
    
    def get_feature_importance(self, plot: bool = False) -> pd.DataFrame:
        """Get feature importances from the trained Random Forest.
        
        Args:
            plot: If True, display a horizontal bar chart of importances.
            
        Returns:
            DataFrame with features and their importance scores, sorted descending.
        """
        if not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importances.")
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        return feature_importance_df
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate the model on held-out test data.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary containing accuracy, precision, recall, F1, and confusion matrix.
        """
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  Confusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained pipeline.
        Default location: REPO_ROOT/models/race_pace_pipeline.pkl
        """
        if path is None:
            out_path = REPO_ROOT / 'models' / 'race_pace_pipeline.pkl'
        else:
            out_path = Path(path)
            
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_path)
        logger.info(f"Pipeline saved to {out_path}")


    def load_model(self, path: Optional[str] = None):
        """
        Load a previously saved pipeline.
        Default location: REPO_ROOT/models/race_pace_pipeline.pkl
        """
        if path is None:
            in_path = REPO_ROOT / 'models' / 'race_pace_pipeline.pkl'
        else:
            in_path = Path(path)

        if not in_path.exists():
            # Fallback for backward compatibility or if user provided relative path
            if not in_path.is_absolute():
                 # Try relative to CWD first, then maybe check old location?
                 # For now, just strict check
                 pass
            raise FileNotFoundError(f"Model file not found at {in_path}")
            
        self.pipeline = joblib.load(in_path)
        logger.info(f"Pipeline loaded successfully from {in_path}")
               
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
    analyzer.save_model()

if __name__ == "__main__":
    main()