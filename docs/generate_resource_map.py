from pathlib import Path


def generate_resource_map():
    current_dir = Path(__file__).parent
    csv_files = list(x.name for x in (current_dir / "csv").glob("*.csv"))
    charts = list(x.name for x in (current_dir / "keyword_analysis").glob("*.html"))
    verification_charts = list(x.name for x in (current_dir / "keyword_analysis" / "verification").glob("*.html"))
    mapping = f'''
const csv_files = {csv_files}  
const charts = {charts} 
const verification_charts = {verification_charts} 
    '''
    with open(current_dir / "resource-paths.js", "w") as f:
        f.write(mapping)


if __name__ == "__main__":
    generate_resource_map()

