import http.client
import csv
import json

# Configuration de la connexion Aviationstack
conn = http.client.HTTPSConnection("api.aviationstack.com")
access_key = "038297fcff9beac5c8c9b985f7c270e0"
endpoint = f"/v1/flights?access_key={access_key}"

# Effectuer une requête pour les données de vols
conn.request("GET", endpoint)
res = conn.getresponse()
data = res.read()

# Décoder les données JSON
flights_data = json.loads(data.decode("utf-8"))

# Vérifier si les données sont récupérées correctement
if "data" not in flights_data:
    print("Aucune donnée disponible !")
    exit()

# Extraire les informations pertinentes
flights_list = flights_data["data"]

# Chemin du fichier CSV
csv_file = r"/Users/loucamalerba/Desktop/IMPORTANT/CentraleSupelec/Cours 2A/Machine Learning Casa/flights_departures_ORY_aviation_stack.csv"


# Écriture dans le fichier CSV
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Écrire l'en-tête
    writer.writerow([
        "Flight Number",
        "Departure Airport",
        "Arrival Airport",
        "Scheduled Departure Time",
        "Actual Departure Time",
        "Scheduled Arrival Time",
        "Actual Arrival Time",
        "Status"
    ])

    # Écrire les données de chaque vol
    for flight in flights_list:
        writer.writerow([
            flight.get("flight", {}).get("iata", ""),  # Numéro de vol
            flight.get("departure", {}).get("airport", ""),  # Aéroport de départ
            flight.get("arrival", {}).get("airport", ""),  # Aéroport d'arrivée
            flight.get("departure", {}).get("scheduled", ""),  # Départ prévu
            flight.get("departure", {}).get("actual", ""),  # Départ réel
            flight.get("arrival", {}).get("scheduled", ""),  # Arrivée prévue
            flight.get("arrival", {}).get("actual", ""),  # Arrivée réelle
            flight.get("flight_status", "")  # Statut du vol
        ])

print(f"Les données des vols ont été enregistrées dans '{csv_file}' avec succès.")
