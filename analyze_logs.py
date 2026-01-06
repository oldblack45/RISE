import pandas as pd
import os

def analyze_logs():
    turn_log_path = r'd:\python_project\RISE\experiments\diplomacy_tournament_20251215_192151\Turn_Log.csv'
    evolution_log_path = r'd:\python_project\RISE\experiments\diplomacy_tournament_20251215_192151\RQ2_Evolution.csv'

    if not os.path.exists(turn_log_path):
        print(f"File not found: {turn_log_path}")
        return

    # 1. Analyze Action Distribution
    print("--- Action Distribution Analysis ---")
    df_turn = pd.read_csv(turn_log_path)
    
    # Filter out empty actions if any
    df_turn = df_turn[df_turn['Action'].notna()]
    
    countries = df_turn['Country'].unique()
    
    for country in countries:
        country_actions = df_turn[df_turn['Country'] == country]
        total_actions = len(country_actions)
        action_counts = country_actions['Action'].value_counts()
        
        print(f"\nCountry: {country} (Total Actions: {total_actions})")
        for action, count in action_counts.items():
            percentage = (count / total_actions) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")

    # 2. Analyze Prediction Accuracy
    if os.path.exists(evolution_log_path):
        print("\n--- Persona Prediction Accuracy Analysis ---")
        df_evo = pd.read_csv(evolution_log_path)
        
        targets = df_evo['Target_Country'].unique()
        
        for target in targets:
            target_data = df_evo[df_evo['Target_Country'] == target]
            accuracy = target_data['Prediction_Accuracy'].mean()
            print(f"Target: {target}, Average Accuracy: {accuracy:.2f}")
            
            # Check trend (first half vs second half)
            mid_point = len(target_data) // 2
            first_half = target_data.iloc[:mid_point]['Prediction_Accuracy'].mean()
            second_half = target_data.iloc[mid_point:]['Prediction_Accuracy'].mean()
            print(f"  First Half: {first_half:.2f}, Second Half: {second_half:.2f}")

if __name__ == "__main__":
    analyze_logs()
