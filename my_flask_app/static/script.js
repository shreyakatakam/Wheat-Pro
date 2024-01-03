function predict() {
    console.log("Predict button clicked");

    const alleleSequence = document.getElementById('allele_sequence').value;
    if (!alleleSequence) {
        console.error('Allele sequence is required.');
        // Handle the error, e.g., display it to the user
        return;
    }
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ alleleSequence }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Prediction error (${response.status}): ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Received prediction data:", data);

        // Redirect to predictions page with query parameters
        const encodedSequence = encodeURIComponent(alleleSequence);
        const encodedData = encodeURIComponent(JSON.stringify(data));
        const predictionsUrl = `/predictions?allele_sequence=${encodedSequence}&data=${encodedData}`;
        window.location.href = predictionsUrl;
    })
    .catch(error => {
        console.error('Prediction error:', error);
        // Handle the error, e.g., display it to the user
    });
}

// Function to navigate to the About Us page
function goToAboutUs() {
    window.location.href = '/about';
}

// Function to navigate to the Contact Us page
function goToContactUs() {
    window.location.href = '/contact';
}
