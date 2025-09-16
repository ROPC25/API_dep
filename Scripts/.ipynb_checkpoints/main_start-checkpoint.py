import subprocess

# Lancer l'API
subprocess.Popen(["python", "API_01_G10.py"])

# Ensuite lancer le dashboard
subprocess.run(["streamlit", "run", "Dashboard_01_G10.py"])
