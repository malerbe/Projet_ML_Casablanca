import pandas as pd
import matplotlib.pyplot as plt


def make_column_hist(df, column, bins=5, title="basic"):
    # Vérifier si la colonne est catégorique ou non
    if df[column].dtype == 'object' or df[column].nunique() < 30:  # On considère un seuil arbitraire pour les catégories
        # Histogramme pour données discrètes (catégories/labels)
        label_counts = df[column].value_counts()
        label_counts = label_counts.sort_index()  # Permet d'organiser les catégories dans l'ordre alphabétique ou numérique
        plt.bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='black')
        if title == "basic":
            plt.title(f'Distribution des labels de {column}')
        else:
            plt.title(title)

        plt.xlabel(f'{column}')
        plt.ylabel('Fréquence')
        plt.xticks(rotation=45, ha='right')  # Rotation des labels pour éviter l'encombrement
    else:
        # Histogramme standard pour données continues
        df[column].hist(bins=bins, color='skyblue', edgecolor='black')
        if title == "basic":
            plt.title(f'Distribution de {column}')
        else:
            plt.title(title)

        plt.xlabel(f'{column}')
        plt.ylabel('Fréquence')

    # Afficher le graphique
    plt.tight_layout()  # Ajuste l'espacement pour que tout s'affiche correctement
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


