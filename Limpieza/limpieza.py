import streamlit as st
import pandas as pd
from io import StringIO 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

st.subheader("Dataset")
data_file = st.file_uploader("Upload CSV",type=["csv"])
columns = {'Perfil Pymetrics', 'Altamente Recomendado', 'Operaciones-Calidad',
       'MTTO-DIMA', 'Comercial-Planeamiento', 'DIGI-SC', 'Resto-Soft',
       'Actividad Grupal.1', 'Apto AG', 'Ingles', 'Apto', 'Destacado',
       'Destacado Pym', 'Ingresados Si/No'}

def mostrarTam(df,state = False):
    st.dataframe(df)
    if not state:
        st.write(f"El dataset sin limpiar tiene las dimensiones de {df.shape[0]} filas y {df.shape[1]} columnas ")
    else:
        st.write(f"El dataset limpio tiene las dimensiones de {df.shape[0]} filas y {df.shape[1]} columnas ")

def graficar(df):
    plt.figure(figsize=(10,6)) 
    sns.displot(
        data=df.isna().melt(value_name="Nulos"), 
        y="variable",
        hue="Nulos",
        multiple="fill",
        aspect=1.25 
    )
    #plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
def limpieza(df):
    df = df.rename(columns={"Fecha Acción":"Fecha Accion", "Año Acción":"Anio Accion", "Periodo Acción":"Periodo Accion", "País":"Pais", 
                   "Tipo de Acción ":"Tipo de Accion", "Acción":"Accion",  "Institución Acción":"Institucion Accion", 
                   "Correo electrónico":"Correo electronico", "Género":"Genero", "Carrera/Titulación":"Carrera/Titulacion", 
                   "Status Académico":"Status Academico", "Inglés":"Ingles_1", "Dirección":"Direccion", "Área":"Area",
                   "Información Real/No Real":"Informacion Real/No Real", "Ingresó":"Ingreso"})
    del df["VIPS"]
    del df["NIPS"]
    # Se crea un DataFrame para delimitar que la información de las personas sólo sean de México
    df = df[df["Pais"] == "México"]
    # Reemplazar NaN por "No"

    # df["Ingresados Si/No"] = df["Ingresados Si/No"].replace(to_replace="NaN", value="No", regex=True,inplace=True)
    df["Ingresados Si/No"] = df["Ingresados Si/No"].replace(np.nan,"No")
    
    # La columna "Actividad Grupal" e "Ingles" se encuentra duplicada y se eliminará la que se encuentra vacía
    del(df["Actividad Grupal"])
    del(df["Ingles_1"])
    del(df["Status Academico"])

    # Se realiza el reemplazo de las variables categóricas que no son necesarias por "0"
    df = df.replace({"False Beginner": 0, "Late hangup": 0})
    
    # Se elimina las columnas que estan vacias
    for col in df:
        if abs(df[col].isna().sum() - df.shape[0]) < 0.1:
            del df[col]

    # Se realiza el reemplazo de las variables categóricas por numéricas
    df["Ingles"] = df["Ingles"].replace({"A1 - Low Beginner": 1, "A2 - High Beginner": 2, "B1 - Low Intermediate": 3, "B2 - High Intermediate": 4, "C1 - Advanced": 5, "C2 - Mastery": 6})
    df["Ingles"] = df["Ingles"].fillna(0)


    # Se reemplazan los valores que se encuentran como caracteres diferentes a los numéricos
    #df["Actividad Grupal.1"] = df["Actividad Grupal.1"].replace(to_replace="No", value=0, regex=True,inplace=True)
    df["Actividad Grupal.1"]= df["Actividad Grupal.1"].map(lambda x:  0 if(x == 'No') else(0 if(x is np.nan) else int(x)))


    # Se cambian Egresado y Egresado/a por 99.
    df['Avance']= df['Avance'].map(lambda x:  99 if(x == 'Egresado' or x == 'Egresado/a') else(0 if(x is np.nan) else x))
    # Se estandariza los datos a numericos pero enteros.
    df['Avance'] = pd.to_numeric(df['Avance'], downcast='signed')
    # Se modifica los datos vacios por 0
    df['Avance']= df['Avance'].map(lambda x:  0 if(math.isnan(x)) else x)

    # Se cambian Egresado y Egresado/a por 99.
    df['Semestres Totales']= df['Semestres Totales'].map(lambda x:  99 if(x == 'Egresado' or x == 'Egresado/a') else(0 if(x is np.nan) else x))

    # Se cambia 12 o más por 13
    df['Semestres Totales']= df['Semestres Totales'].map(lambda x:  13 if(x == '12 o más') else(0 if(x is np.nan) else int(x)))

    # Se estandariza los datos a numericos pero enteros.

    # Se modifica los datos vacios por 0
    df['Semestres Totales']= df['Semestres Totales'].map(lambda x:  0 if(math.isnan(x)) else(0 if(x is np.nan) else int(x)))



    # Nueva columna con la que trabajaremos.
    new_column = []
    # Se analiza ambas columnas para poder determinar el status academico del egresado del candidato
    # según las condiciones establecidas.
    for Av, To in zip(df["Avance"], df["Semestres Totales"]):
        if (Av != 0 and(Av - To == 0 or Av == 99 )):
            new_column.append('Egresado')
        elif(Av - To != 0):
            new_column.append('En curso')
        else:
            new_column.append('Indefinido')
    
    # Crear Situación Academica y guardarlo en new column
    df["Situacion Academica"] = new_column
   
    # Convertimos de float64 a Int64
    df['Anio Accion'] = df['Anio Accion'].astype('Int64')
    df['ID Evento'] = df['ID Evento'].astype('Int64')
    df['ID Candidato'] = df['ID Candidato'].astype('Int64')

    df['ID Evento'] = df['ID Evento'].fillna(0)
    df['ID Candidato'] = df['ID Candidato'].fillna(0)


    new_col = []
    for prim, seg in zip(df["Apto/No Apto"], df["Destacado Pym"]):
        if prim == "Apto" and seg == "Destacado":
            new_col.append("Destacado")

        elif prim == "Apto" and type(seg) == float:
            new_col.append("Apto")

        elif prim == "No Apto" and type(seg) == float:
            new_col.append("No Apto")

        else:
            new_col.append(" ")
    
    # Los datos de la nueva columna se asignan a la columna nombrada como "Apto/No Apto"
    df["Apto/No Apto"] = new_col
    # Se elimina la columna de "Destacado Pym"
    del df["Destacado Pym"]
    # Se renombra la columna "Apto/No Apto"
    df.rename(columns = {"Apto/No Apto" : "Apto/ No Apto/ Destacado"}, inplace = True)
    
    # Se realiza el reemplazo de las variables categóricas por numéricas
    df["Apto/ No Apto/ Destacado"]= df["Apto/ No Apto/ Destacado"].map(lambda x:  1 if x == 'Apto' else(2 if(x=='Destacado') else 0))

    df["Apto AG"]= df["Apto AG"].map(lambda x:  "No Apto" if(x is np.nan) else "Apto")
    df["Apto"]= df["Apto"].map(lambda x:  "No Apto" if(x is np.nan) else "Apto")
    df["Destacado"]= df["Destacado"].map(lambda x:  "No" if(x is np.nan) else "Si")
    df["Evaluados Si/No"]= df["Evaluados Si/No"].map(lambda x:  "No" if(x is np.nan) else "Si")
    df["Postulados Si/No"]= df["Postulados Si/No"].map(lambda x:  "No" if(x is np.nan) else "Si")
    df["Altamente Recomendado"]= df["Altamente Recomendado"].map(lambda x:  "No" if(x is np.nan) else "Si")
    df["Perfil Pymetrics"]= df["Perfil Pymetrics"].map(lambda x:  "0" if(x is np.nan) else x)
    df["Referente Ternium"]= df["Referente Ternium"].map(lambda x:  "No tiene" if(x is np.nan) else x)
    df["Nacionalidad"]= df["Nacionalidad"].map(lambda x:  "No tiene" if(x is np.nan) else x)
    df["Carrera/Titulacion"]= df["Carrera/Titulacion"].map(lambda x:  "Indefinido" if(x is np.nan) else x)
    df["Especialidad"]= df["Especialidad"].map(lambda x:  "Indefinido/No cuenta" if(x is np.nan) else x)


    df["Operaciones-Calidad"]= df["Operaciones-Calidad"].map(lambda x:  "Do Not Recommend" if(x is np.nan) else x)
    df["MTTO-DIMA"]= df["MTTO-DIMA"].map(lambda x:  "Do Not Recommend" if(x is np.nan) else x)
    df["Comercial-Planeamiento"]= df["Comercial-Planeamiento"].map(lambda x:  "Do Not Recommend" if(x is np.nan) else x)
    df["DIGI-SC"]= df["DIGI-SC"].map(lambda x:  "Do Not Recommend" if(x is np.nan) else x)
    df["Resto-Soft"]= df["Resto-Soft"].map(lambda x:  "Do Not Recommend" if(x is np.nan) else x)
    
    return df

if data_file is not None:
    file_details = {"filename":data_file.name, "filetype":data_file.type,"filesize":data_file.size}

    df = pd.read_csv(data_file)

    if(columns.issubset(df.columns)):
        graficar(df)
        mostrarTam(df)
        if st.button('Realizar la limpieza'):
            df = limpieza(df)
            mostrarTam(df,True)
            graficar(df)
            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='Ternium.csv',
                mime='text/csv',
            )
    else:
        st.write("Introduzca un archivo valido.")