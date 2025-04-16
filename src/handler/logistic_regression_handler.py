import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionHandler:
    def __init__(self, test_size, random_state):
        self.dependent_variable = "CONTAMINACAO_INDICE"
        self.independent_variables = None
        self.test_size = test_size
        self.random_state = random_state

    def logistic_regression(self, df_dataset):
        df_concatenado = df_dataset.drop('MUNICIPIO', axis=1)
        variaveis = df_concatenado.columns.tolist()

        variaveis_independentes = variaveis.copy()
        variaveis_independentes.remove('MEDIA_PONDERADA_INDICE')

        X = df_concatenado[variaveis_independentes]
        y = df_concatenado[self.dependent_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        model = LogisticRegression(solver='lbfgs', max_iter=1000)

        # Treinar o modelo
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        print("Acurácia:", accuracy_score(y_test, y_pred))
        print("Relatório de classificação:\n", classification_report(y_test, y_pred))

        # Previsões no conjunto de teste
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        # Matriz de confusão
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusão")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()

        print("Pare aqui")




