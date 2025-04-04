import streamlit as st
import pandas as pd
import calendar
import pulp
from pulp import PULP_CBC_CMD
import io
import logging
import warnings

warnings.filterwarnings("ignore")

# Configuration du logging : les logs seront enregistrés dans debug.log
logging.basicConfig(filename="debug.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s: %(message)s")
logging.info("Application démarrée.")

st.title("Optimisation de répartition des kilomètres (Objective)")
st.write("Chargez votre fichier d'entrée et définissez les paramètres.")

# --- Barre latérale ---
st.sidebar.header("Paramètres de la date")
ref_date = st.sidebar.date_input("Date de référence", value=pd.to_datetime("2025-01-01"))
year = ref_date.year
month = ref_date.month
days_in_month = calendar.monthrange(year, month)[1]
jour_du_mois = ref_date.day
jours_restants = days_in_month - jour_du_mois
st.sidebar.write(f"Jours restants dans le mois : {jours_restants}")

# Champs pour Δ_min et Δ_max (en km/jour)
delta_min_day = st.sidebar.number_input("Δ_min (km/jour)", value=100.0, step=10.0, format="%.0f")
delta_max_day = st.sidebar.number_input("Δ_max (km/jour)", value=550.0, step=10.0, format="%.0f")

# Champ pour le prix carburant
fuel_price = st.sidebar.number_input("Prix Carburant (MAD/L)", value=20.0, step=0.1, format="%.2f")
st.sidebar.write(f"Prix Carburant actuel : {fuel_price} MAD/L")

# Sélecteur pour le mode de calcul du total mensuel
mode_total = st.sidebar.selectbox("Mode de calcul du total mensuel", options=["Automatique", "Manuel"])
if mode_total == "Manuel":
    total_mois_manuel = st.sidebar.number_input("Total kilométrage prévu pour le mois", value=100000.0, step=1000.0, format="%.0f")

# Bloc de sliders pour la répartition par palier
st.sidebar.header("Répartition par palier (%)")
p0 = st.sidebar.slider("Palier [0 - 4000]", 0, 100, 20, 1)
p1 = st.sidebar.slider("Palier [4000 - 8000]", 0, 100, 20, 1)
p2 = st.sidebar.slider("Palier [8000 - 11000]", 0, 100, 20, 1)
p3 = st.sidebar.slider("Palier [11000 - 14000]", 0, 100, 20, 1)
p4 = st.sidebar.slider("Palier (>14000)", 0, 100, 20, 1)
total_pourc = p0 + p1 + p2 + p3 + p4
st.sidebar.write(f"Somme des pourcentages : {total_pourc} %")
if total_pourc != 100:
    st.sidebar.error("La somme des pourcentages doit être égale à 100 %.")
    st.stop()

# --- Chargement du fichier d'entrée ---
uploaded_file = st.file_uploader("Choisissez le fichier Excel d'entrée (colonnes : Transporteur, Immatriculation, Total)", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        logging.info("Fichier d'entrée chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        logging.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

    st.write("### Aperçu du fichier d'entrée")
    st.dataframe(df.head())
    
    logging.info(f"Colonnes du fichier : {df.columns.tolist()}")
    if "Total" not in df.columns:
        st.error("La colonne 'Total' n'a pas été trouvée dans le fichier.")
        logging.error(f"Colonne 'Total' introuvable. Colonnes disponibles: {df.columns.tolist()}")
        st.stop()
    
    # Une fois le fichier chargé, afficher la liste des matricules pour saisie manuelle
    matricules = df["Immatriculation"].unique()
    selected_matricules = st.sidebar.multiselect("Sélectionnez les matricules pour saisie manuelle", options=matricules)
    
    # Saisie manuelle pour les matricules sélectionnés
    manual_totals = {}
    if selected_matricules:
        st.sidebar.write("Saisissez le total mensuel pour chaque camion sélectionné :")
        for matricule in selected_matricules:
            total_manual = st.sidebar.number_input(f"Total mensuel pour {matricule}", value=0.0, step=1000., format="%.0f")
            manual_totals[matricule] = total_manual

    total_deja = df["Total"].sum()
    st.write(f"**Total déjà parcouru** = {total_deja}")

    # Calcul du total mensuel selon le mode choisi
    if mode_total == "Automatique":
        moyenne_journaliere = total_deja / jour_du_mois if jour_du_mois > 0 else 0
        total_mois = total_deja + round(jours_restants * moyenne_journaliere)
    else:
        total_mois = total_mois_manuel
    st.write(f"**Total mensuel** = {total_mois}")

    R = total_mois - total_deja
    st.write(f"**Km restants à répartir** = {R}")

    # Calcul de Δ_min et Δ_max sur l'ensemble des jours restants
    min_km_par_camion = jours_restants * delta_min_day
    max_km_par_camion = jours_restants * delta_max_day

    # Extraire les camions avec saisie manuelle
    if selected_matricules:
        df_manual = df[df["Immatriculation"].isin(selected_matricules)].copy()
        df_manual["Total Mensuel Saisi"] = df_manual["Immatriculation"].map(manual_totals)
        df_manual["Km Restants (Manuel)"] = df_manual["Total Mensuel Saisi"] - df_manual["Total"]
        # Pour chaque camion manuel, déduire l'intervalle et calculer le tarif
        # Nous utilisons ici les mêmes intervalles que pour l'optimisation
        def determine_interval(total_final):
            if total_final <= 4000:
                return "[0 - 4000]", 0
            elif total_final <= 8000:
                return "[4000 - 8000]", 1
            elif total_final <= 11000:
                return "[8000 - 11000]", 2
            elif total_final <= 14000:
                return "[11001 - 14000]", 3
            else:
                return ">14000", 4

        # Pour les camions manuels, "Total Finale" sera le total mensuel saisi
        df_manual["Total Finale"] = df_manual["Total Mensuel Saisi"]
        intervals = []
        tarifs_manual = []
        for idx, row in df_manual.iterrows():
            interval_str, palier_index = determine_interval(row["Total Finale"])
            intervals.append(interval_str)
            # Récupérer le tarif mis à jour pour ce transporteur et ce palier
            prest = row["Transporteur"]
            t_val = None
            if prest in manual_totals:  # cela ne devrait pas se produire
                t_val = manual_totals[prest]
            else:
                # On utilise le dictionnaire des tarifs mis à jour (qui sera défini ci-dessous)
                pass  # Nous définirons ensuite "tarifs" pour le modèle
            tarifs_manual.append(None)  # Nous allons les calculer ensuite
        df_manual["Intervalle Palier"] = intervals
        # Nous ne calculerons pas encore le tarif pour les manuels ici, car nous allons utiliser la même logique que pour l'optimisation
        # (Nous calculerons cela dans la suite du script, après mise à jour des tarifs)

        st.write("### Camions avec saisie manuelle")
        st.dataframe(df_manual)
    else:
        df_manual = pd.DataFrame()

    # Filtrer les camions non manuels pour l'optimisation
    df_opt = df[~df["Immatriculation"].isin(selected_matricules)].copy()
    if not df_opt.empty:
        df_used = df_opt.copy()
    else:
        df_used = df.copy()
    
    # Réindexer df_used pour itérer facilement
    df_used.reset_index(drop=True, inplace=True)

    # --- Définition des paliers et intervalles ---
    L = [0, 4000, 8000, 11000, 14001]
    U = [4000, 8000, 11000, 14000, 999999]
    num_paliers = 5
    palier_intervals = {
        0: "[0 - 4000]",
        1: "[4000 - 8000]",
        2: "[8000 - 11000]",
        3: "[11001 - 14000]",
        4: ">14000"
    }

    # Données tarifaires de référence pour chaque prestataire
    tarif_data = [
        {'PRESTATAIRE': 'COMPTOIR SERVICE', 'KM': '[0 - 4000]', 'A fixe': 4.2, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'COMPTOIR SERVICE', 'KM': '[4000-8000]', 'A fixe': 4.2, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'COMPTOIR SERVICE', 'KM': '[8000-11000]', 'A fixe': 3.4, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'COMPTOIR SERVICE', 'KM': '[11000-14000]', 'A fixe': 3.2, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'COMPTOIR SERVICE', 'KM': '>14000', 'A fixe': 3.2, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'SDTM', 'KM': '[0 - 4000]', 'A fixe': 4.58, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'SDTM', 'KM': '[4000-8000]', 'A fixe': 4.58, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'SDTM', 'KM': '[8000-11000]', 'A fixe': 4.16, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'SDTM', 'KM': '[11000-14000]', 'A fixe': 3.65, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'SDTM', 'KM': '>14000', 'A fixe': 3.18, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'TRANSMEL SARL', 'KM': '[0 - 4000]', 'A fixe': 3.25, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'TRANSMEL SARL', 'KM': '[4000-8000]', 'A fixe': 4.26, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'TRANSMEL SARL', 'KM': '[8000-11000]', 'A fixe': 4.26, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'TRANSMEL SARL', 'KM': '[11000-14000]', 'A fixe': 3.73, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'TRANSMEL SARL', 'KM': '>14000', 'A fixe': 3.25, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'S.T INDUSTRIE', 'KM': '[0 - 4000]', 'A fixe': 3.25, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'S.T INDUSTRIE', 'KM': '[4000-8000]', 'A fixe': 4.26, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'S.T INDUSTRIE', 'KM': '[8000-11000]', 'A fixe': 4.26, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'S.T INDUSTRIE', 'KM': '[11000-14000]', 'A fixe': 3.73, 'Quote part gasoil': 0.35},
        {'PRESTATAIRE': 'S.T INDUSTRIE', 'KM': '>14000', 'A fixe': 3.25, 'Quote part gasoil': 0.35}
    ]

    def normalize_interval(s):
        return s.replace(" ", "").lower()

    ref_intervals = ["[0-4000]", "[4000-8000]", "[8000-11000]", "[11000-14000]", ">14000"]

    # Calcul des tarifs mis à jour : tarif = A fixe + (Quote part gasoil * fuel_price)
    updated_tarifs = {}
    for prest in df_used["Transporteur"].unique():
        prest_tarifs = [None] * 5
        for d in tarif_data:
            if d["PRESTATAIRE"].lower() == prest.lower():
                interval_norm = normalize_interval(d["KM"])
                if interval_norm in ref_intervals:
                    idx = ref_intervals.index(interval_norm)
                    prest_tarifs[idx] = d["A fixe"] + d["Quote part gasoil"] * fuel_price
        updated_tarifs[prest] = prest_tarifs
    logging.info(f"Tarifs mis à jour avec le prix carburant {fuel_price} : {updated_tarifs}")
    tarifs = updated_tarifs

    # Optimisation sur df_used
    N = len(df_used)
    trucks = range(N)
    paliers = range(num_paliers)

    eligible_trucks = [i for i in trucks if df_used.loc[i, "Total"] + max_km_par_camion >= 8000]
    logging.info(f"{len(eligible_trucks)} camions éligibles sur {N} pour atteindre 8000 km.")

    if st.button("Lancer l'optimisation"):
        if total_pourc != 100:
            st.error("La somme des pourcentages doit être égale à 100%.")
            logging.error("La somme des pourcentages n'est pas égale à 100%.")
            st.stop()

        # Variables de déviation pour les contraintes souples
        lambda_penalty = 10000
        s_plus = pulp.LpVariable.dicts("s_plus", paliers, lowBound=0, cat=pulp.LpContinuous)
        s_minus = pulp.LpVariable.dicts("s_minus", paliers, lowBound=0, cat=pulp.LpContinuous)

        with st.spinner("Optimisation en cours (limite 60 s)..."):
            try:
                model = pulp.LpProblem("Optimisation_km", pulp.LpMinimize)

                x = pulp.LpVariable.dicts("x", trucks, lowBound=0, cat=pulp.LpContinuous)
                Delta = pulp.LpVariable.dicts("Delta", trucks,
                                              lowBound=min_km_par_camion,
                                              upBound=max_km_par_camion,
                                              cat=pulp.LpContinuous)
                for i in trucks:
                    km_deja = df_used.loc[i, "Total"]
                    model += x[i] == km_deja + Delta[i], f"Def_x_{i}"
                model += pulp.lpSum(Delta[i] for i in trucks) == R, "Total_Delta"

                y = pulp.LpVariable.dicts("y", (trucks, paliers), cat=pulp.LpBinary)
                z = pulp.LpVariable.dicts("z", (trucks, paliers), lowBound=0, cat=pulp.LpContinuous)
                for i in trucks:
                    model += x[i] == pulp.lpSum(z[i][j] for j in paliers), f"Decoupage_x_{i}"
                for i in trucks:
                    model += pulp.lpSum(y[i][j] for j in paliers) == 1, f"Unique_palier_{i}"

                M = 999999
                for i in trucks:
                    km_deja = df_used.loc[i, "Total"]
                    for j in paliers:
                        LB_ij = max(km_deja, L[j])
                        model += z[i][j] >= LB_ij * y[i][j], f"LB_{i}_{j}"
                        model += z[i][j] <= U[j] * y[i][j] + M*(1 - y[i][j]), f"UB_{i}_{j}"
                        if km_deja > U[j]:
                            model += y[i][j] == 0, f"Exclure_{i}_{j}"

                for i in trucks:
                    model += x[i] <= 4000 + M*(1 - y[i][0]), f"Max_x_palier0_{i}"
                    model += x[i] >= 4000 * y[i][1], f"Min_x_palier1_{i}"
                    model += x[i] <= 8000 + M*(1 - y[i][1]), f"Max_x_palier1_{i}"
                    model += x[i] >= 8000 * y[i][2], f"Min_x_palier2_{i}"
                    model += x[i] <= 11000 + M*(1 - y[i][2]), f"Max_x_palier2_{i}"
                    model += x[i] >= 11000 * y[i][3], f"Min_x_palier3_{i}"
                    model += x[i] <= 14000 + M*(1 - y[i][3]), f"Max_x_palier3_{i}"
                    model += x[i] >= 14001 * y[i][4], f"Min_x_palier4_{i}"

                # Contraintes souples sur la répartition : pour chaque palier j, cible T_j = (p_j/100)*N
                for j in paliers:
                    T_j = ([p0, p1, p2, p3, p4][j] / 100.0) * N
                    model += (pulp.lpSum(y[i][j] for i in trucks) - T_j) == (s_plus[j] - s_minus[j]), f"Deviat_palier_{j}"

                model += (
                    pulp.lpSum(
                        tarifs[df_used.loc[i, "Transporteur"]][j] * z[i][j]
                        for i in trucks for j in paliers
                    )
                    + lambda_penalty * pulp.lpSum(s_plus[j] + s_minus[j] for j in paliers)
                ), "Cout_Total_et_Deviations"

                solver = PULP_CBC_CMD(msg=1, timeLimit=60)
                model.solve(solver=solver)
                status = pulp.LpStatus[model.status]
                logging.info(f"Status du solveur : {status}")

            except Exception as e:
                st.error(f"Erreur lors de l'optimisation : {e}")
                logging.error(f"Erreur d'optimisation : {e}")
                st.stop()

        if status in ["Optimal", "Not Solved"]:
            st.success("Optimisation terminée !")
            recaps = []
            for j in paliers:
                count = sum(pulp.value(y[i][j]) for i in trucks)
                recaps.append({
                    "Palier": palier_intervals[j],
                    "Nombre de camions": count,
                    "Pourcentage (%)": (count / N) * 100
                })
            recap_df = pd.DataFrame(recaps)
            st.write("### Récapitulatif de répartition par palier")
            st.dataframe(recap_df)
            logging.info(f"Récapitulatif de répartition: {recap_df.to_dict(orient='records')}")

            resultats = []
            for i in trucks:
                transporteur = df_used.loc[i, "Transporteur"]
                immatriculation = df_used.loc[i, "Immatriculation"]
                km_deja = df_used.loc[i, "Total"]
                x_val = pulp.value(x[i])
                delta_val = pulp.value(Delta[i])
                assigned_palier = None
                intervalle = None
                tarif_val = None
                for j in paliers:
                    if pulp.value(y[i][j]) > 0.5:
                        assigned_palier = j
                        intervalle = palier_intervals[j]
                        tarif_val = tarifs[transporteur][j]
                        break
                resultats.append({
                    "Immatriculation": immatriculation,
                    "Transporteur": transporteur,
                    "Total": km_deja,
                    "Variation": delta_val,
                    "Total Finale": x_val,
                    "Intervalle Palier": intervalle,
                    "Tarif (MAD/km)": tarif_val
                })
            
            df_resultats = pd.DataFrame(resultats)
            ligne_total = {
                "Immatriculation": "Total",
                "Transporteur": "",
                "Total": df_resultats["Total"].sum(),
                "Variation": df_resultats["Variation"].sum(),
                "Total Finale": df_resultats["Total Finale"].sum(),
                "Intervalle Palier": "",
                "Tarif (MAD/km)": round(pulp.value(model.objective), 2)
            }
            df_resultats = pd.concat([df_resultats, pd.DataFrame([ligne_total])], ignore_index=True)

            # Pour les camions manuels, calculer l'intervalle et le tarif selon leur Total Finale
            if not df_manual.empty:
                manual_rows = []
                for idx, row in df_manual.iterrows():
                    # Pour un camion manuel, le Total Finale est celui saisi par l'utilisateur
                    total_finale = row["Total Mensuel Saisi"]
                    # Calculer l'intervalle correspondant
                    if total_finale <= 4000:
                        interval_str, palier_idx = "[0 - 4000]", 0
                    elif total_finale <= 8000:
                        interval_str, palier_idx = "[4000 - 8000]", 1
                    elif total_finale <= 11000:
                        interval_str, palier_idx = "[8000 - 11000]", 2
                    elif total_finale <= 14000:
                        interval_str, palier_idx = "[11001 - 14000]", 3
                    else:
                        interval_str, palier_idx = ">14000", 4
                    # Récupérer le tarif mis à jour pour ce prestataire et ce palier
                    prest = row["Transporteur"]
                    tarif_manual = tarifs[prest][palier_idx] if prest in tarifs and tarifs[prest][palier_idx] is not None else None

                    manual_rows.append({
                        "Immatriculation": row["Immatriculation"],
                        "Transporteur": row["Transporteur"],
                        "Total": row["Total"],
                        "Variation": row["Km Restants (Manuel)"],
                        "Total Finale": total_finale,
                        "Intervalle Palier": interval_str,
                        "Tarif (MAD/km)": tarif_manual
                    })
                df_manual_calc = pd.DataFrame(manual_rows)
            else:
                df_manual_calc = pd.DataFrame()

            # Concaténer les résultats d'optimisation et les résultats manuels
            final_df = pd.concat([df_resultats, df_manual_calc], ignore_index=True)
            # Si vous souhaitez que les camions manuels apparaissent AVANT la ligne Total, on peut trier :
            final_df = final_df[final_df["Immatriculation"] != "Total"]
            # Calculer à nouveau la ligne Total sur l'ensemble combiné
            total_line = {
                "Immatriculation": "Total",
                "Transporteur": "",
                "Total": final_df["Total"].sum(),
                "Variation": final_df["Variation"].sum(),
                "Total Finale": final_df["Total Finale"].sum(),
                "Intervalle Palier": "",
                "Tarif (MAD/km)": ""
            }
            final_df = pd.concat([final_df, pd.DataFrame([total_line])], ignore_index=True)

            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False, sheet_name="Optimisation")
                recap_df.to_excel(writer, index=False, sheet_name="Répartition")
            towrite.seek(0)
            st.download_button(
                label="Télécharger le fichier optimisé",
                data=towrite,
                file_name="resultats_optimisation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Aucune solution optimale n'a été trouvée dans le délai imparti ou le problème est infaisable.")
            logging.error("Problème infaisable ou temps dépassé.")
else:
    st.info("Veuillez charger un fichier Excel pour commencer.")
