import camelot



file = r'foo.pdf'
# tables = camelot.read_pdf(file, flavor='stream', table_areas=['5,510,600,380'], row_tol=5)
tables = camelot.read_pdf(file)

tables
tables.export(file, f='csv', compress=True)
tables[0]
tables[0].parsing_report
tables[0].to_csv('1.csv')
tables[0].df


camelot.plot(tables[0], kind='grid').show()