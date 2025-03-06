function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for (let radio of uiBathrooms) {
        if (radio.checked) {
            return parseInt(radio.value);
        }
    }
    return -1; // Invalid Value
}

function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for (let radio of uiBHK) {
        if (radio.checked) {
            return parseInt(radio.value);
        }
    }
    return -1; // Invalid Value
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft").value;
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations").value;
    var estPrice = document.getElementById("uiEstimatedPrice");

    if (!location) {
        alert("Please select a location.");
        return;
    }
    var url = "/api/predict_home_price";
    //var url = "http://127.0.0.1:5000/predict_home_price";

    $.post(url, {
        total_sqft: parseFloat(sqft),
        size_numeric: parseInt(bhk),
        bath: parseInt(bathrooms),
        location: location
    }, function(data, status) {
        console.log("Response received:", data);
        
        if (!data || !data.estimated_price) {
            console.error("Error: estimated_price is undefined or missing in response.");
            estPrice.innerHTML = "<h2>Error: Price not available</h2>";
            estPrice.style.color = "red";
            return;
        }

        estPrice.innerHTML = `<h2>Estimated Price: ₹ ${data.estimated_price} Lakh</h2>`;
        estPrice.style.color = "#28a745";
    }).fail(function(xhr, status, error) {
        console.error("API Request Failed: ", error);
        alert("Error fetching price. Check console.");
    });
}

function onPageLoad() {
    console.log("Document loaded");
    var url = "/api/get_location_names";
    //var url = "http://127.0.0.1:5000/get_location_names";

    $.get(url, function(data, status) {
        console.log("Got response for get_location_names", data);
        if (data && data.locations) {
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            $('#uiLocations').append(new Option("Choose a Location", "", true, true));

            for (let i = 0; i < data.locations.length; i++) {
                $('#uiLocations').append(new Option(data.locations[i], data.locations[i]));
            }
        }
    }).fail(function(xhr, status, error) {
        console.error("Error fetching locations:", error);
        alert("Failed to fetch locations. Check console.");
    });
}

window.onload = onPageLoad;
