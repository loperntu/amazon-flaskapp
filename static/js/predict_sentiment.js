window.onload = function(){
  function predictSentiment() {
  
    var inputText = document.getElementById("textbox").value;
    var selectedModel = document.getElementById('ml-model').value;
    var predictionContainer = document.getElementById("prediction");
    var predictionText = document.getElementById("prediction-text");
    var originalText = document.getElementById("original-text")
  
    predictionContainer.style.display = 'flex';
    predictionText.style.display= 'None'
  
    var serverData=[{
      'text':inputText,
      'mlmodel':selectedModel
    }]

    $.post({
      url:"/predict", 
      data:JSON.stringify(serverData), 
      contentType:'application/json'}).done(function (data){
        predictionText.style.display = 'flex';
        predictionText.textContent = data;
        originalText.textContent = inputText;
      }
    );
  
    //  $.get("/predict?"+"text="+inputText+"&"+"mlmodel="+selectedModel).done(function (data) {
    //    predictionText.innerHTML = data;
    //  })
  }
  
  var button = document.getElementById('button')
  button.addEventListener('click', predictSentiment)
}
