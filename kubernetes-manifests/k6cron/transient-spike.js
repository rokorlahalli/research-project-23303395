import http from 'k6/http';

export const options = {
  vus: 20,          // number of concurrent virtual users
  duration: '15s',  // run for 15 seconds
};

export default function () {
  http.get('http://catalogue.app.svc.cluster.local:8080/health');
  http.get('http://user.app.svc.cluster.local:8080/health');
  http.get('http://cart.app.svc.cluster.local:8080/health');
  http.get('http://web.app.svc.cluster.local:8080');
}
