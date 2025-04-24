document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('interest-form');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const interestError = document.getElementById('interest-error');
    const MAX_SELECTIONS = 3;
    
    // Fetch available interests from backend
    fetch('/interests')
        .then(response => response.json())
        .then(data => {
            const interestsContainer = document.getElementById('interests-container');
            const interests = data.interests;
            
            // Create buttons for each interest
            interests.forEach(interest => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'interest-btn';
                button.textContent = interest;
                button.dataset.interest = interest;
                
                button.addEventListener('click', function() {
                    // If button is not selected and we haven't reached max selections
                    if (!this.classList.contains('selected')) {
                        if (document.querySelectorAll('.interest-btn.selected').length >= MAX_SELECTIONS) {
                            interestError.classList.remove('d-none');
                            return;
                        }
                    }
                    
                    this.classList.toggle('selected');
                    interestError.classList.add('d-none');
                });
                
                interestsContainer.appendChild(button);
            });
        })
        .catch(error => {
            console.error('Error fetching interests:', error);
        });
    
    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedButtons = document.querySelectorAll('.interest-btn.selected');
        if (selectedButtons.length === 0) {
            interestError.textContent = 'Please select at least 1 interest.';
            interestError.classList.remove('d-none');
            return;
        }
        
        if (selectedButtons.length > MAX_SELECTIONS) {
            interestError.textContent = 'Please select up to 3 interests.';
            interestError.classList.remove('d-none');
            return;
        }
        
        const selectedInterests = Array.from(selectedButtons).map(btn => btn.dataset.interest).join(', ');
        
        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ interests: selectedInterests })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                errorDiv.textContent = data.error;
                errorDiv.classList.remove('d-none');
                resultsDiv.classList.add('d-none');
            } else {
                document.getElementById('processed-interests').textContent = data.processed_input;
                document.getElementById('recommended-rewards').textContent = data.recommendations.join(', ');
                document.getElementById('tier').textContent = data.tier;
                resultsDiv.classList.remove('d-none');
                errorDiv.classList.add('d-none');
                interestError.classList.add('d-none');
            }
        })
        .catch(error => {
            errorDiv.textContent = 'An error occurred while fetching recommendations.';
            errorDiv.classList.remove('d-none');
            resultsDiv.classList.add('d-none');
        });
    });
});