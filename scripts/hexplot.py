#!/usr/bin/env python

import sys
import subprocess
import tempfile
import os
from charis.hexplot import create_hexplot

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: hexplot <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    
    # Create a temporary directory for our Bokeh application
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the main application file
        app_path = os.path.join(temp_dir, 'app.py')
        with open(app_path, 'w') as f:
            f.write(f"""
from charis.hexplot import create_hexplot
create_hexplot('{filename}')
""")
        
        # Run bokeh serve with the application
        subprocess.run(
            ["bokeh", "serve", "--show", app_path],
            check=True,
            stdout=subprocess.PIPE
        )

if __name__ == "__main__":
    main()
