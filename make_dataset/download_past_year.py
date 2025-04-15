import requests
import csv
import datetime
from dateutil.relativedelta import relativedelta

# URL de base pour l'API avec votre clé
BASE_URL = "https://aviation-edge.com/v2/public/flightsHistory"
API_KEY = "4db1d6-90e68e"


for airport_code in ["CDG", "BVA"]:
    # Nom du fichier CSV final
    CSV_FILENAME = f"flights_history_last_12_months_{airport_code}.csv"

    # Calculer les périodes pour les 12 derniers mois
    end_date = datetime.date(2025, 4, 14) - datetime.timedelta(days=3)# Date d'aujourd'hui
    start_date = end_date - relativedelta(months=12) + datetime.timedelta(days=4)  # 12 mois avant

    # Préparation du fichier CSV
    with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)

        # Écrire les en-têtes (colonnes du fichier CSV)
        csv_writer.writerow([
            "Flight Number",
            "Airline Name",
            "Airline IATA Code",
            "Airline ICAO Code",
            "Departure Airport IATA",
            "Departure Airport ICAO",
            "Departure Terminal",
            "Departure Gate",
            "Scheduled Departure Time",
            "Estimated Departure Time",
            "Actual Departure Time",
            "Arrival Airport IATA",
            "Arrival Airport ICAO",
            "Scheduled Arrival Time",
            "Estimated Arrival Time",
            "Flight Status"
        ])

        # Diviser les 12 derniers mois en intervalles de 31 jours maximum
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + datetime.timedelta(days=15), end_date)

            # Formater les dates au format attendu par l'API
            date_from = current_start.strftime("%Y-%m-%d")
            date_to = current_end.strftime("%Y-%m-%d")

            # Construire l'URL pour cette période
            url = f"{BASE_URL}?code={airport_code}&type=departure&date_from={date_from}&date_to={date_to}&key={API_KEY}"

            # Faire une requête à l'API
            print(f"Récupération des vols de {date_from} à {date_to}...")
            response = requests.get(url)

            if response.status_code == 200:
                flights_data = response.json()

                if not flights_data:
                    print(f"Aucune donnée disponible pour la période {date_from} à {date_to}.")
                else:
                    # Parcourir chaque vol et écrire les données dans le fichier CSV

                    for flight in flights_data:
                        csv_writer.writerow([
                            flight.get("flight", {}).get("iataNumber", ""),                # Numéro de vol IATA
                            flight.get("airline", {}).get("name", ""),                     # Nom de la compagnie aérienne
                            flight.get("airline", {}).get("iataCode", ""),                 # Code IATA de la compagnie
                            flight.get("airline", {}).get("icaoCode", ""),                 # Code ICAO de la compagnie
                            flight.get("departure", {}).get("iataCode", ""),               # Aéroport de départ IATA
                            flight.get("departure", {}).get("icaoCode", ""),               # Aéroport de départ ICAO
                            flight.get("departure", {}).get("terminal", ""),               # Terminal de départ
                            flight.get("departure", {}).get("gate", ""),                   # Porte d'embarquement
                            flight.get("departure", {}).get("scheduledTime", ""),          # Horaire prévu de départ
                            flight.get("departure", {}).get("estimatedTime", ""),          # Horaire estimé de départ
                            flight.get("departure", {}).get("actualTime", ""),             # Horaire réel de départ
                            flight.get("arrival", {}).get("iataCode", ""),                 # Aéroport d'arrivée IATA
                            flight.get("arrival", {}).get("icaoCode", ""),                 # Aéroport d'arrivée ICAO
                            flight.get("arrival", {}).get("scheduledTime", ""),            # Horaire prévu d'arrivée
                            flight.get("arrival", {}).get("estimatedTime", ""),            # Horaire estimé d'arrivée
                            flight.get("status", "")                                       # Statut du vol
                        ])
                    print(f"Données enregistrées pour la période {date_from} à {date_to}.")
            else:
                print(f"Erreur lors de la requête API pour la période {date_from} à {date_to}. Code HTTP : {response.status_code}")

            # Passer à la prochaine période de 31 jours
            current_start = current_end + datetime.timedelta(days=1)

    print(f"Les données des 12 derniers mois ont été enregistrées dans '{CSV_FILENAME}' avec succès.")

