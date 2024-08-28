import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.eval()

# Define a function to perform image classification
def predict(image):
    # Apply transformations to the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(output.data, 1)
        predicted_class = predicted_class.item()

    # Return the predicted class and probability
    return predicted_class, probabilities[predicted_class]

# Define a function to compute the performance metrics of the model
def performance_metrics():
    # Load the test data
    test_data = torch.load("test_data.pth", map_location=torch.device('cpu'))

    # Get the predictions and ground truth labels for the test data
    all_predictions = []
    all_labels = []
    for images, labels in test_data:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                _, predicted_class = torch.max(output.data, 1)
                all_predictions.append(predicted_class.item())
                all_labels.append(label.item())

    # Compute the performance metrics
    target_names = ['Acnitic Keratosis','Dermatofibroma','Vascular Lesion']
    report = classification_report(all_labels, all_predictions, target_names=target_names)
    confusion = confusion_matrix(all_labels, all_predictions)

    # Return the performance metrics
    return report, confusion

# Define the main function to run the app
def main():
    # Add a title to the app
    st.title("Image Classification App")

    # Add an uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # If an image has been uploaded
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Add a button to predict the class
        if st.button("Predict Class"):
            # Call the predict function to make predictions
            predicted_class, probability = predict(image)

            # Display the predicted class and probability
            st.write("Predicted Class:", predicted_class)
            st.write("Probability:", probability.item())

        # Add a button to compute the performance metrics
        if st.button("Compute Performance Metrics"):
            # Call the performance_metrics function to compute the performance metrics
            report, confusion = performance_metrics()

            # Display the performance metrics
            st.write("Classification Report:")
            st.write(report)
            st.write("Confusion Matrix:")
            st.write(confusion)

# Call the main function to run the app
if __name__ == '__main__':
    main()
