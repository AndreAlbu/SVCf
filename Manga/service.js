window.onload = function() {
  // Obter a URL atual
  const currentUrl = window.location.href;

  // Selecionar todos os links da navegação
  const links = document.querySelectorAll("nav a");

  // Iterar sobre todos os links
  links.forEach(link => {
      // Verificar se o href do link corresponde à URL atual
      if (currentUrl.includes(link.href)) {
        link.classList.add("selected"); // Adiciona a classe 'selected'
        loadHTML(`${link.id}.html`, "main");
      }
  });
};

function previewImage(event) {
    const file = event.target.files[0]; // Pegando o primeiro arquivo
    const reader = new FileReader(); // Criando um FileReader para ler a imagem

    reader.onload = function() {
        const imgElement = document.getElementById('imagePreview');
        imgElement.src = reader.result; // Definindo a URL da imagem
        imgElement.style.display = 'block'; // Mostrando a imagem
    }

    if (file) {
        reader.readAsDataURL(file); // Lendo a imagem como URL de dados
    }
}

// Função para carregar um arquivo HTML em um div
function loadHTML(file, elementId) {
  fetch(file)
    .then(response => response.text())
    .then(data => {
        document.getElementById(elementId).innerHTML = data;
    })
    .catch(error => console.error('Erro ao carregar o arquivo:', error));
}