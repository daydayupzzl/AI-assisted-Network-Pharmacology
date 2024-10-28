import pandas as pd

from src.tools.dataset_repurpose import create_dataset_for_repurpose
from src.tools.utils import *


def repurpose(X_repurpose, target, model, drug_names=None, device=None, target_name=None, dataset=None, model_st=None,
              traindata=None, version=None, batch_size=500, num_workers=4):
    directory = os.path.join('case_study', dataset, f"{traindata}_{model_st}_{version}")
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"repurpose_{traindata}_{model_st}_{target_name}.xlsx")

    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping save.")
        return

    try:
        test_data = create_dataset_for_repurpose(X_repurpose, drug_names, target, dataset, target_name, model_st,
                                                 version)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers
        )
        measured, y_pred = predicting(model, device, test_loader)
        y_pred = y_pred.tolist()
    except Exception as e:
        print(f"Error during dataset creation or prediction: {e}")
        return

    print_list = []
    if drug_names is not None:
        f_d = max([len(o) for o in drug_names]) + 1
        for i in range(len(X_repurpose)):
            string_lst = [drug_names[i], target_name, f"{y_pred[i]:.2f}"]
            print_list.append((string_lst, y_pred[i]))

    print_list.sort(key=lambda x: x[1], reverse=True)
    sorted_data = [i[0] for i in print_list]

    df = pd.DataFrame(sorted_data, columns=['DrugName', 'TargetName', 'BindingScore'])
    df.insert(0, 'Rank', range(1, len(df) + 1))

    try:
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            worksheet.set_column('A:A', 10)
            worksheet.set_column('B:B', 45)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:D', 12)
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return

    print(f"Results saved to {file_path}")

def create_output_directory(dataset, traindata, model_st, version):
    directory = os.path.join('case_study', dataset, f"{traindata}_{model_st}_{version}")
    os.makedirs(directory, exist_ok=True)
    return directory

def save_results_to_excel(file_path, sorted_data):
    df = pd.DataFrame(sorted_data, columns=['DrugName', 'TargetName', 'BindingScore'])
    df.insert(0, 'Rank', range(1, len(df) + 1))

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column('A:A', 10)
        worksheet.set_column('B:B', 45)
        worksheet.set_column('C:C', 15)
        worksheet.set_column('D:D', 12)