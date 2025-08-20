import datasets

# Load a specific year and split
def assemble_datasets(years, cik):
    dataset_list = []
    for year in years:
        year_input = "year_" + str(year)
        dataset_year = datasets.load_dataset("eloukas/edgar-corpus", year_input, split="train",  trust_remote_code=True)
        dataset_filtered = dataset_year.filter(lambda x: x["cik"] == cik)
        dataset_list.append(dataset_filtered)
    dataset = datasets.concatenate_datasets(dataset_list)
    return dataset

filings_years = range(2016, 2020 + 1, 1)
company_cik = "5272"  # AIG's CIK
aig_filings = assemble_datasets(filings_years, company_cik)
print(aig_filings)

aig_filings.save_to_disk("aig_filings")