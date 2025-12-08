from typing import List
import shutil
from pathlib import Path
import csv
import ast
from tqdm.auto import tqdm

def refresh_directories(dir_paths: List[str]):
    for dir_path in dir_paths:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_staff_list(csv_file: str) -> List[str]:
    staff_list = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            paper_staff = ast.literal_eval(row['list_of_staff'].strip())
            staff_list.extend(paper_staff)
    return list(set(staff_list))

def get_author_for_pdf(pdf_id: str) -> List[str]:
    with open('research_paper_unique.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['paper_id']) == int(pdf_id):
                paper_staff = ast.literal_eval(row['list_of_staff'].strip())
                return paper_staff
    return []

def get_uic_staff_details(csv_file: str) -> List[dict]:
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        uic_staff: list[dict] = []
        uic_staff_unique: set[str] = set()
        progress_bar = tqdm(total=sum(1 for _ in open(csv_file, 'r', encoding='utf-8')) - 1, desc="Processing staff entries")
        for row in reader:
            staff_name_1 = row['staff_name1'].strip()
            staff_name_2 = row['staff_name2'].strip()
            staff_name_3 = row['staff_name3'].strip()

            staff_dept_1 = row['staff_dept1'].strip()
            staff_dept_2 = row['staff_dept2'].strip()
            staff_dept_3 = row['staff_dept3'].strip()

            staff_title_1 = row['staff_title1'].strip()
            staff_title_2 = row['staff_title2'].strip()
            staff_title_3 = row['staff_title3'].strip()

            if staff_name_1 and staff_name_1 not in uic_staff_unique:
                uic_staff.append({
                    'name': staff_name_1,
                    'department': staff_dept_1,
                    'title': staff_title_1
                })
                uic_staff_unique.add(staff_name_1)

            if staff_name_2 and staff_name_2 not in uic_staff_unique:
                uic_staff.append({
                    'name': staff_name_2,
                    'department': staff_dept_2,
                    'title': staff_title_2
                })
                uic_staff_unique.add(staff_name_2)

            if staff_name_3 and staff_name_3 not in uic_staff_unique:
                uic_staff.append({
                    'name': staff_name_3,
                    'department': staff_dept_3,
                    'title': staff_title_3
                })
                uic_staff_unique.add(staff_name_3)

            progress_bar.update(1)
        progress_bar.close()

            
    return uic_staff

def get_year(page_id: str) -> str:
    with open('research_paper_unique.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['paper_id']) == int(page_id):
                return row['year'].strip()
    return "Unknown"

def get_paper_link(page_id: str) -> str:
    with open('research_paper_unique.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['paper_id']) == int(page_id):
                return row['paper_link'].strip()
    return ""
