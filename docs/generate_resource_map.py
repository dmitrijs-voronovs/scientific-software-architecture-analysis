from pathlib import Path


def generate_resource_map():
    current_dir = Path(__file__).parent
    csvFiles = list(x.name for x in (current_dir / "csv").glob("*.csv"))
    charts = list(x.name for x in (current_dir / "keyword_analysis").glob("*.html"))
    validationCharts = list(x.name for x in (current_dir / "keyword_analysis" / "verification").glob("*.html"))
    map = f'''
const csvFiles = {csvFiles}  
const charts = {charts} 
const validationCharts = {validationCharts} 
    '''
    with open(current_dir / "resource-paths.js", "w") as f:
        f.write(map)


if __name__ == "__main__":
    generate_resource_map()

