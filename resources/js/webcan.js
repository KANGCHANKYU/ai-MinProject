const socket = new WebSocket("ws://127.0.0.1:8000/ws");
const devVideo = document.getElementById('video-container')
let imgElement;
// 웹소켓 연결 시
socket.onopen = function(event) {
  console.log("서버와 연결됨");
};

socket.onmessage = async function(event) {
  // JSON 데이터 수신
  const data = JSON.parse(event.data); // 이미지 데이터를 받음
  console.log(data)
  if(data.hasOwnProperty("result")){
    console.log(data.result)
  }
  else if(data.hasOwnProperty("image")){
    console.log(data)
    if (!imgElement) {
      imgElement = document.createElement('img');
      devVideo.appendChild(imgElement);
    }
    console.log(data)
    // 이미지 데이터를 Blob으로 변환
    const blob = base64ToBlob(data.image);
    const imageBlob = new Blob([blob], { type: 'image/jpeg' });
    
    // 받은 이미지를 화면에 표시 (이미지 엘리먼트의 src 변경)
    const imageURL = URL.createObjectURL(imageBlob);
    imgElement.src = imageURL;
  }
};

function base64ToBlob(base64String) {
  const byteCharacters = atob(base64String);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray]);
}
   
// 웹소켓 오류 시
socket.onerror = function(error) {
  console.error(`웹소켓 오류: ${error}`);
};

// 웹소켓 연결 종료 시
socket.onclose = function(event) {
  if (event.wasClean) {
    console.log(`웹소켓 연결이 깨끗하게 종료되었습니다. 코드: ${event.code}, 이유: ${event.reason}`);
  } else {
    console.error(`웹소켓 연결이 종료되었습니다.`);
  }
};
