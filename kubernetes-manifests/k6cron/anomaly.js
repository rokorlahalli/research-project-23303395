import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 25 },
    { duration: '1m', target: 75 },
    { duration: '30s', target: 0 },
  ],
};

export default function () {
  http.get('http://catalogue.app.svc.cluster.local:8080/health');
  http.get('http://user.app.svc.cluster.local:8080/health');
  http.get('http://cart.app.svc.cluster.local:8080/health');
  sleep(1);
}