#Gabriela Masak
#D598: Analytics Programming
#Task 2

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

#Read in data from Excel file
businessData = pd.read_excel(r'C:\Users\gabri\PycharmProjects\pythonProject\D598 Data Set.xlsx')
#print(businessData)

#Remove duplicates
businessDataUpdated = businessData.drop_duplicates(subset = ['Business ID'], keep = 'first', inplace=False)
#print(businessDataUpdated)

#Group businesses by state, calculate descriptive statistics on fields
debtStatsByState = businessDataUpdated.groupby('Business State').agg({'Total Long-term Debt': ['mean', 'median', 'min', 'max']})
#print(debtStatsByState)
equityStatsByState = businessDataUpdated.groupby('Business State').agg({'Total Equity': ['mean', 'median', 'min', 'max']})
debtToEquityStatsByState = businessDataUpdated.groupby('Business State').agg({'Debt to Equity': ['mean', 'median', 'min', 'max']})
liabilitiesStatsByState = businessDataUpdated.groupby('Business State').agg({'Total Liabilities': ['mean', 'median', 'min', 'max']})
revenueStatsByState = businessDataUpdated.groupby('Business State').agg({'Total Revenue': ['mean', 'median', 'min', 'max']})
profitMarginStatsByState = businessDataUpdated.groupby('Business State').agg({'Profit Margin': ['mean', 'median', 'min', 'max']})

#Merge into singular dataframe
businessStatsByState = pd.concat([debtStatsByState, equityStatsByState, debtToEquityStatsByState, liabilitiesStatsByState, revenueStatsByState, profitMarginStatsByState], join = 'outer', axis=1)
print(businessStatsByState)

#Identify businesses with negative debt-to-equity ratios
negativeDebtEquity = businessDataUpdated[businessDataUpdated['Debt to Equity'] < 0]
#print(negativeDebtEquity)
print(negativeDebtEquity['Business ID'])

#Create dataframe with businesses' calculated debt-to-income ratios and concatenate:
businessDataUpdated['Debt to Income Ratio'] = businessDataUpdated['Total Long-term Debt'] / businessDataUpdated['Total Revenue']
#New dataset with only debt-to-income ratios:
debtToIncomeRatio = businessDataUpdated['Debt to Income Ratio']
#print(debtToIncomeRatio)
print(businessDataUpdated)
