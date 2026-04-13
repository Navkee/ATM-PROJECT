import mlflow

# Simple function

def calculate_bill(item_cost, tax_rate):
	tax_amount = item_cost * tax_rate
	total = item_cost + tax_amount + 40  # delivery charge
	return total

# Start MLflow run

mlflow.start_run()

# Inputs

cost = 500
tax = 0.05

# Log inputs

mlflow.log_param("item_cost", cost)
mlflow.log_param("tax_rate", tax)

# Calculate

final_bill = calculate_bill(cost, tax)

# Log output

mlflow.log_metric("total_bill_amount", final_bill)

# End run

mlflow.end_run()

print(f"Run tracked successfully! Total bill is: {final_bill}")
