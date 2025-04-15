import pandas as pd
import matplotlib.pyplot as plt


def make_column_hist(df, column, bins=5, title="basic"):
    df[column].hist(bins=bins, color='skyblue', edgecolor='black')
    if title == "basic":
        plt.title(f'Distribution de {column}')
    else:
        plt.title(title)

    plt.xlabel(f'{column}')
    plt.ylabel('Fréquence')
    plt.show()


def camembert_colonne(dataframe, colonne, titre='Camembert des Catégories'):
    """
    Fonction pour créer un camembert à partir des données d'une colonne.
    Les labels et fréquences sont extraits automatiquement.
    
    Arguments :
    ----------
    - dataframe : pd.DataFrame, le DataFrame contenant les données.
    - colonne : str, le nom de la colonne à analyser.
    - titre : str, titre du graphique (optionnel).
    """
    # Calcul des fréquences pour chaque catégorie
    valeurs = dataframe[colonne].value_counts()
    labels = valeurs.index  # Les labels sont les valeurs uniques de la colonne

    # Création du camembert
    plt.pie(valeurs, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(titre)
    plt.show()


