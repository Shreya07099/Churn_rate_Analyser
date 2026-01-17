import shap 
import numpy as np
import matplotlib.pyplot as plt
from Churn_rate_Analyser.Final_model import clf_xgb, x_train, x_test
#Using the first ten customers from the test data set to provide explainations for(any one)
def quick_explain_tree(model,features,x_test_sample):
    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(x_test_sample[0:10])
        # 1. Summary plot (most important)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_test_sample[:10], 
                     feature_names=features, show=True)
    plt.title("Feature Impact on Churn Prediction")
    plt.tight_layout()
    plt.title("Feature Impact on Churn Prediction")
    plt.tight_layout()
    plt.close()
    # Choose which customer you want business insights for, and print feature score
    customer_1_shap = shap_values[0]
    top_indices = np.argsort(-np.abs(customer_1_shap))[:10]
    print("Shap Explainations output")
    for idx in top_indices:
        feature = features[idx]
        impact = customer_1_shap[idx]
        direction = "INCREASES" if impact > 0 else "REDUCES"
        print(f"• {feature}: {direction} churn risk by {abs(impact):.2%}")
    #import numpy as np
    print()
    print("INDIVIDUAL CUSTOMER CHURN ANALYSIS")
    print()

    # Choose which customer to analyze (0 = first, 1 = second, 2 = third)
    customer_index = 0 # Change this to 0, 1, or 2

    # Get probability for ONE specific customer
    single_customer_data = x_test_sample[customer_index:customer_index+1]  # Get just this one
    prob = model.predict_proba(single_customer_data)[:, 1][0]  # Single probability
    print(f" CHURN PROBABILITY: {prob:.2%} ({prob:.4f})")
    return shap_values, explainer

shap_values, explainer = quick_explain_tree(clf_xgb, x_train.columns.tolist(),x_test)
'''print(shap_values)
print(explainer)'''
