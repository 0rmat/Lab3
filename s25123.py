import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Ładowanie danych
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
df = pd.read_csv(url)

# Eksploracja danych
print(df.head())
print(df.info())
print(df.describe())

# Liczba unikalnych wartości dla każdej kolumny
print("Unikalne wartosci w kazdej kolumnie:\n", df.nunique())

# Rozkład zmiennych
plt.figure(figsize=(12, 8))
df.hist(bins=30, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle('Rozkład zmiennych')
plt.tight_layout()
plt.savefig('variable_distribution.png')
plt.show()

# Analiza korelacji dla kolumn numerycznych
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Wybór tylko kolumn numerycznych
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Macierz korelacji')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()


# Rozkład zmiennej score
plt.figure(figsize=(8, 5))
sns.histplot(df['score'], bins=30, kde=True, color='blue')
plt.title('Rozkład zmiennej score')
plt.xlabel('score')
plt.ylabel('Częstotliwość')
plt.grid(axis='y')
plt.savefig('score_distribution.png')
plt.show()

# Rozkład score według płci
plt.figure(figsize=(8, 5))
sns.boxplot(x='gender', y='score', hue='gender', data=df, palette='coolwarm', legend=False)
plt.title('Rozkład zmiennej score według płci')
plt.xlabel('Płeć')
plt.ylabel('Score')
plt.grid(True)
plt.savefig('score_by_gender.png')
plt.show()

# Analiza brakujących danych
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
print(f"\nBrakujace dane:\n{missing_data}\n")
print(f"\nProcent brakujacych danych:\n{missing_percentage}")

# Procentowe wystąpienie zmiennych kategorycznych
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nProcentowe wystapienie wartosci w kolumnie {col}:")
    print(df[col].value_counts(normalize=True) * 100)

# Kodowanie zmiennych kategorycznych
df_encoded = pd.get_dummies(df, drop_first=True)

# Przygotowanie danych
X = df_encoded.drop(columns=['score'])
y = df_encoded['score']

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Trenowanie modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Predykcja i ocena modelu
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nSredni blad kwadratowy (MSE): {mse:.2f}')
print(f'Wspolczynnik determinacji R^2: {r2:.2f}')
print(f'Sredni blad absolutny (MAE): {mae:.2f}')

# Wizualizacja rzeczywistych vs przewidywanych wartości
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.title('Rzeczywiste vs Przewidywane wartości')
plt.grid(True)
plt.savefig('predicted_vs_actual.png')
plt.show()

# Wizualizacja pozostałości modelu
plt.figure(figsize=(8, 5))
residuals = y_test - y_pred
sns.histplot(residuals, bins=30, kde=True, color='red')
plt.title('Rozkład pozostałości')
plt.xlabel('Pozostałości')
plt.ylabel('Częstotliwość')
plt.grid(axis='y')
plt.savefig('residuals_distribution.png')
plt.show()

# Zapis wyników do pliku .md
with open('extended_results_report.md', 'w', encoding='utf-8') as doc:
    doc.write('# Raport z analizy danych o college\'ach\n\n')
    doc.write('## Wprowadzenie\n')
    doc.write(
        'Celem tej analizy jest badanie danych o college\'ach i prognozowanie wyników na podstawie różnych czynników.\n\n')

    doc.write('## Analiza danych\n')
    doc.write(f'Calkowita liczba rekordow: {len(df)}\n')
    doc.write(f'Brakujace wartosci: {df.isnull().sum().to_dict()}\n\n')
    doc.write(f'Procent brakujacych danych:\n{missing_percentage}\n\n')
    doc.write('### Statystyki opisowe:\n')
    doc.write(f'{df.describe()}\n\n')

    doc.write('## Wizualizacje\n')
    doc.write('- Rozklad zmiennych: ![Rozkład zmiennych](variable_distribution.png)\n')
    doc.write('- Macierz korelacji: ![Macierz korelacji](correlation_matrix.png)\n')
    doc.write('- Rozkład zmiennej score: ![Rozkład zmiennej score](score_distribution.png)\n')
    doc.write('- Rozkład score według płci: ![Rozkład score według płci](score_by_gender.png)\n')
    doc.write('- Rzeczywiste vs Przewidywane wartości: ![Rzeczywiste vs Przewidywane wartości](predicted_vs_actual.png)\n')
    doc.write('- Rozkład pozostałości: ![Rozkład pozostałości](residuals_distribution.png)\n\n')

    doc.write('## Model\n')
    doc.write(f'Sredni blad kwadratowy (MSE): {mse:.2f}\n')
    doc.write(f'Wspolczynnik determinacji R²: {r2:.2f}\n')
    doc.write(f'Sredni blad absolutny (MAE): {mae:.2f}\n\n')

    doc.write('## Wnioski\n')
    doc.write('Wnioski oparte na wynikach modelu mogą być dodane tutaj.\n')

print("Raport zostal zapisany w pliku extended_results_report.md.")