import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the dataset (replace 'vehicle_details.csv' with your actual dataset file)
df = pd.read_csv('Vehiclebdata.csv')
spareparts_df = pd.read_csv('spareparts.csv')  # Load spare parts dataset

# Combine relevant features for text processing
df['combined_features'] = df['How was the issue observed'] + ' ' + df['What is the issue'] + ' ' + df[
    'Why is this an issue']

# Create TF-IDF vectorizer and transform existing data
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df['combined_features'])


# Function to predict the output columns based on input
def predict_issue(observation, issue, issue_reason):
    input_text = f"{observation} {issue} {issue_reason}"
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, feature_matrix)
    most_similar_index = similarities.argmax()  # Get the most similar record
    similarity_score = similarities[0][most_similar_index]

    # Retrieve the details from the most similar record
    prediction = df.iloc[most_similar_index]

    return prediction, similarity_score


# Streamlit interface
def home_page():
    st.title("Vehicle Issue Prediction Tool")
    st.image("zoo.png", use_column_width=True)  # Add an image to the home page

    # Get user input for the issue description
    observation = st.text_input("How was the issue observed:")
    issue = st.text_input("What is the issue:")
    issue_reason = st.text_input("Why is this an issue:")

    if st.button("Predict"):
        if observation and issue and issue_reason:
            # Get prediction based on input
            prediction, similarity_score = predict_issue(observation, issue, issue_reason)

            # Display predicted results
            st.subheader("Predicted Issue Details")
            st.write("Ticket Number:", prediction["Ticket Number"])
            st.write("Root cause:", prediction["Root cause"])
            st.write("Root cause category:", prediction["Root cause category"])
            st.write("Solution implemented:", prediction["Solution implemented"])
            st.write("Team solved:", prediction["Team solved"])
            st.write("Time Required (hrs):", prediction["Time Required (hrs)"])
            st.write("Total Cost (USD):", prediction["Total Cost (USD)"])
            st.write("Cosine similarity score:", f"{similarity_score:.2f}")

            # Show warning if the similarity score is low
            if similarity_score < 0.5:
                st.warning("Low similarity score. The prediction may not be accurate.")
        else:
            st.error("Please fill out all fields.")


def service_details_page():
    st.title("Vehicle Service History")
    st.image("service history.jpeg", use_column_width=True)  # Add an image to the service details page

    vehicle_id = st.text_input("Enter Vehicle ID (e.g., V01, V02, etc.):")

    if st.button('Get Service Details'):
        # Get service details based on Vehicle ID
        service_data = df[df['Vehicle ID'] == vehicle_id]

        if not service_data.empty:
            service_data_sorted = service_data.sort_values(by='date', ascending=False)
            st.write("Last Service Details:")
            last_service = service_data_sorted.iloc[0]

            st.write(f"Ticket Number: {last_service['Ticket Number']}")
            st.write(f"Root cause: {last_service['Root cause']}")
        else:
            st.write("No service record found for this vehicle.")


def technicians_page():
    st.title("Technician Ratings Page")
    st.image("technicians.jpeg", use_column_width=True)  # Add an image to the technicians page

    technician_ratings = df.groupby('Technician ID')['Service Rating'].mean().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(x='Technician ID', y='Service Rating', data=technician_ratings, palette='Set2', ax=ax)
    ax.set_title('Technician Ratings')
    st.pyplot(fig)


def ticket_info_page():
    st.title("Ticket Info Page")
    st.image("ticket info.jpeg", use_column_width=True)  # Add an image to the ticket info page

    ticket_status = ['Open', 'Closed', 'Open', 'Closed', 'Open']
    ticket_counter = Counter(ticket_status)

    fig, ax = plt.subplots()
    ax.pie(ticket_counter.values(), labels=ticket_counter.keys(), autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
    ax.set_title('Open vs Closed Tickets')
    st.pyplot(fig)


def spare_parts_info_page():
    st.title("Spare Parts Information")
    st.image("service.jpeg", use_column_width=True)
    # Get Part ID input from the user
    part_id = st.text_input("Enter Part ID:")

    if st.button('Get Spare Part Info'):
        # Get spare part details based on Part ID
        part_data = spareparts_df[spareparts_df['Part ID'] == part_id]

        if not part_data.empty:
            # Display the spare part details
            st.write("Part Details:")
            st.write(f"Part Name: {part_data['Part Name'].values[0]}")
            st.write(f"Category: {part_data['Category'].values[0]}")
            st.write(f"Supplier ID: {part_data['Supplier ID'].values[0]}")
            st.write(f"Stock Quantity: {part_data['Stock Quantity'].values[0]}")
            st.write(f"Minimum Stock Level: {part_data['Minimum Stock Level'].values[0]}")
            st.write(f"Unit Price (USD): {part_data['Unit Price (USD)'].values[0]}")
            st.write(f"Warranty Period (Months): {part_data['Warranty Period (Months)'].values[0]}")
            st.write(f"Part Manufacturer: {part_data['Part Manufacturer'].values[0]}")
            st.write(f"Model Compatibility: {part_data['Model Compatibility'].values[0]}")
        else:
            st.write("No details found for this Part ID.")


def search_issues_page():
    st.title("Search Issues by Keyword")
    st.image("issue search.jpeg", use_column_width=True)  # Add an image to the search issues page

    keyword = st.text_input("Enter a keyword to search for related issues:")

    if st.button('Search'):
        if keyword:
            matching_issues = df[df['combined_features'].str.contains(keyword, case=False, na=False)]

            if not matching_issues.empty:
                st.subheader(f"Issues related to '{keyword}'")
                for index, row in matching_issues.iterrows():
                    st.write(f"Ticket Number: {row['Ticket Number']}")
                    st.write(f"Issue Details: {row['What is the issue']}")
                    st.write("---")
            else:
                st.write(f"No issues found related to the keyword: {keyword}")
        else:
            st.error("Please enter a keyword to search.")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:",
                            ["Home", "Service Details", "Technicians", "Ticket Info", "Spare Parts Info", "Search Issues"])

    if page == "Home":
        home_page()
    elif page == "Service Details":
        service_details_page()
    elif page == "Technicians":
        technicians_page()
    elif page == "Ticket Info":
        ticket_info_page()
    elif page == "Spare Parts Info":
        spare_parts_info_page()
    elif page == "Search Issues":
        search_issues_page()


if __name__ == '__main__':
    main()
