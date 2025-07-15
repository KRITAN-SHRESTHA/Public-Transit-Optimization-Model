from ortools.linear_solver import pywraplp
import math
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox

def optimize_bus_routes(DD,D,R,S,C1,C2,N1,N2,Cost1,Cost2, ST, M):

    w = [math.ceil(length / M) for length in ST] # Calculate trips per shift (rounding up)
    SP=[0.4,0.2,0.35,0.05]
    # Create the solver SCIP

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Impossible to create the Solver.")
        exit()

    # --- Variables ---
    x = {}
    y = {}
    buses_type1 = {}  # Number of buses Type I
    buses_type2 = {}  # Number of buses Type II
    for i in range(R):
        for j in range(S):
            x[i, j] = solver.IntVar(0, solver.infinity(), f'x_{i}_{j}') # x[i][j]: Trips of Type I Bus on route i, shift j
            y[i, j] = solver.IntVar(0, solver.infinity(), f'y_{i}_{j}') # y[i][j]: Trips of Type II Bus on route i, shift j
            buses_type1[i, j] = solver.IntVar(0, solver.infinity(), f'buses_type1_{i}_{j}')  # Buses Tipo I
            buses_type2[i, j] = solver.IntVar(0, solver.infinity(), f'buses_type2_{i}_{j}')  # Buses Tipo II

    # Splitting Data into Demand and TripFactor
 #   D = data.iloc[:, 1:5]  #  # Extract columns D1 to D4 (Indexes 1 to 4) Demand
    T = data.iloc[:, 5:9]  # Extract columns T1 to T4 (Indexes 5 to 8) Trip Factor
    Pr=data.iloc[:, 9:10] # Extract columns Route Damand Portion (Indexes 5 to 8) Trip Factor
    D = pd.DataFrame([[round(DD * pr * SP[j]) for j in range(S)] for pr in Pr.iloc[:, 0]], columns=[f"D{j+1}" for j in range(S)])  # Calculate D[i,j] as ceiling of DD * Pr[i] * SP[j]
    
    # Define output file path
    output_file_pathD = r"C:\Users\Wilson\GUSCanada\Group2 Operations Analytics Project - Group2\Deliverables\DemandPr.csv"
    Pr.to_csv(output_file_pathD, index=False)
    # Constraint: Non-Negativity is also handled by IntVar(0, solver.infinity())

    # --- Objective Function ---
    objective = solver.Sum(Cost1*x[i, j] + Cost2*y[i, j] for i in range(R) for j in range(S)) # Minimize Cost of Trips in a day
    solver.Minimize(objective)

    # --- Constraints ---
    # 1. Demand Satisfaction
    for i in range(R):
        for j in range(S):
            solver.Add(C1*x[i, j] + C2*y[i, j] >= D.iloc[i,j])
    
    # 2. Capacity: Fleet Availability for Bus Type-I + Type-II
    for j in range(S):
            solver.Add(
                solver.Sum(buses_type1[i, j] + buses_type2[i, j] for i in range(R)) <= (N1 + N2)
            )
    # 3. Capacity: Fleet Availability for Bus Type-I (Max N1)
    for j in range(S):
            solver.Add(
                solver.Sum(buses_type1[i, j] for i in range(R)) <= N1
            )

    # 4. Capacity: Fleet Availability for Bus Type-II (Max N2)
    
    for j in range(S):
            solver.Add(
                solver.Sum(buses_type2[i, j] for i in range(R)) <= N2
            )
    # 5. Customer Experience: Minimum Service Level for low demand routes
    for i in range(R):
        for j in range(S):
            solver.Add(x[i, j] + y[i, j] >= w[j])

    # 6. Constraint Trip and Bus Relation to ensure Reuse of Buses during shift

    for i in range(R):
        for j in range(S):
            T_factor = T.iloc[i, j] # T_factor = min(T.iloc[i, j], w[j])
            # For Tipo I: buses_type1[i, j] >= x[i, j] / capacity
            solver.Add(buses_type1[i, j] * T_factor >= x[i, j])
            # For Tipo II: buses_type2[i, j] >= y[i, j] / capacity
            solver.Add(buses_type2[i, j] * T_factor >= y[i, j])


    # Solve the model
    status = solver.Solve()


    if status == pywraplp.Solver.OPTIMAL:     # Check if a solution was found
        print("\nOptimal solution found!")
    
        total_cost = solver.Objective().Value()  # Optimized Cost from Solver
        
    #--- Extracting results into a DataFrame ---
        # Generate column names dynamically
        column_names = ["Route"]
        for i in range(1, S + 1):
            column_names.append(f"Shift {i} - # Trips Bus T-1")
            column_names.append(f"Shift {i} - # Trips Bus T-2")
            column_names.append(f"Shift {i} - # Buses T-1")
            column_names.append(f"Shift {i} - # Buses T-2")

        # Collect all results in a single DataFrame
        results = []
        Total_cost=0
        for i in range(R):
            row_data = [f"Route {i+1}"]
            for j in range(S):
                t1_trips = int(x[i, j].solution_value())
                t2_trips = int(y[i, j].solution_value())
                buses_T1 = int(buses_type1[i, j].solution_value()) 
                buses_T2 = int(buses_type2[i, j].solution_value())
                row_data.extend([t1_trips, t2_trips, buses_T1, buses_T2])
            results.append(row_data)


        # Create the final output DataFrame
        results_df = pd.DataFrame(results, columns=column_names)

        # Append the totals row (sum for each numeric column)
        totals = results_df.iloc[:, 1:].astype(int).sum()
        totals_row = pd.DataFrame([["TOTAL"] + totals.tolist()], columns=column_names)

        # Append the totals row to the results dataframe
        results_df = pd.concat([results_df, totals_row], ignore_index=True)
        return results_df, total_cost
    else:
        print("\nNo optimal solution found.")
        return None


# Define the file path
file_path = r"C:\Users\Wilson\GUSCanada\Group2 Operations Analytics Project - Group2\Deliverables\Pr.csv"


data = pd.read_csv(file_path) # Load the CSV Demand file into a pandas DataFrame
R=93 #total Routes
S=4 # daily Shifts
C1=60 #Capacity Bus Type I
C2=90 #Capacith Bus Type II
N1=600 # Fleet of Buses Type I
N2=90 # Fleet of Buses Type II
Cost1=200 # Weight to priorize by Bus Type-I (Updated)
Cost2=300 # Weight to priorize by Bus Type-II (Updated)
ST=[195,360,240,90] # Shifts Time Length in Minutes [Shift 1, Shift 2, Shift 3, Shift 4]
M=60 # MWT: Max Wait Time: 30 Mins
F=5 #Fare

#Ask Demand D
root = tk.Tk()
root.withdraw()
DD = simpledialog.askinteger("Input Demand", "Enter daily demand:", minvalue=0)
if DD is None:
    messagebox.showerror("Error", "Demand input canceled. Exiting.")
    DD = 0  # Default Value
root.destroy()

# Define output file path
output_file_path = r"C:\Users\Wilson\GUSCanada\Group2 Operations Analytics Project - Group2\Deliverables\G2Results.csv"

# Call function with default parameters
final_df, total_cost = optimize_bus_routes(DD,data,R,S,C1,C2,N1,N2,Cost1,Cost2, ST, M)

# Save to CSV
if final_df is not None:  # Valid DataFrame?
    final_df.to_csv(output_file_path, index=False)
    print(final_df.head())  # Display the first few rows of output
    print(final_df.tail())  # Display the last few rows of output
    print(f"Total Optimized Cost: {total_cost}")
    totals_row = final_df.iloc[-1]  # Last row contains totals  
    # Calculate sums from the columns in final_df
    total_trips_t1 = totals_row[[col for col in final_df.columns if 'Trips Bus T-1' in col]].sum()
    total_trips_t2 = totals_row[[col for col in final_df.columns if 'Trips Bus T-2' in col]].sum()
    total_buses_t1 = totals_row[[col for col in final_df.columns if 'Buses T-1' in col]].sum()
    total_buses_t2 = totals_row[[col for col in final_df.columns if 'Buses T-2' in col]].sum()
    total_trips = total_trips_t1 + total_trips_t2
    total_buses = total_buses_t1 + total_buses_t2

    # Create summary table
    summary_data = {
        "Metric": ["Total Trips T1", "Total Trips T2", "Total Buses T1", "Total Buses T2", 
                   "Total Trips", "Total Buses","Frequency (Mins)", "Total Cost","Daily Riders Demand","Fare","Revenue"],
        "Value": [total_trips_t1, total_trips_t2, total_buses_t1, total_buses_t2, 
                  total_trips, total_buses,M, total_cost,DD,F,F*DD]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Save and display summary
    print("\nSummary Table:")
    print(summary_df)
else:
    print("No results to save or display.")