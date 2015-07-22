'use strict';

var express      = require('express'),
    path         = require('path'),
    app          = express(),
    MongoClient  = require('mongodb').MongoClient,
    bodyParser   = require('body-parser'),
    accept       = require('http-accept'),
    fs           = require('fs'),
    ini          = require('ini');

var contentPath = path.resolve(__dirname + '/build/'), port = 5000;
var env = process.env.NODE_ENV || 'development';
var dataPath = __dirname + '/data/';
var config = ini.parse(fs.readFileSync(dataPath + 'mongodb.ini', 'utf8'));
var labels = JSON.parse(fs.readFileSync(dataPath + 'labels.json', 'utf8'));
var labelId = 'labels';

app.use(accept);
app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json());
app.use(express.static(contentPath));

app.set('views', contentPath + '/jade');
app.set('view engine', 'jade');


app.get('/api/:collection', function (request, response) {
   MongoClient.connect('mongodb://' + config.mongodb.client_host + '/' + config.mongodb.db_name, function (err, db) {
      var coll = db.collection(request.params.collection);
      var query = {};
      query[labelId] = null;
      coll.count({}, function (err, total) {
         coll.count(query, function (err, count) {
            var rand = Math.floor(Math.random() * count);
            coll.findOne(query, {skip: rand}, function (err, cluster) {
               db.close();
               
               var tags = {top10: [], bottom10: []}, l = cluster.ctweettuplelist.length;
               if (l > 19) {
                  for (var i = 0; i < 10; i++) tags.top10.push({text: cluster.ctweettuplelist[i][2]});
                  for (var j = l - 10; j < l; j++) tags.bottom10.push({text: cluster.ctweettuplelist[j][2]});
               } else {
                  cluster.ctweettuplelist.forEach(function (tuple, index) {
                     var toPush = index < 10 ? 'top10' : 'bottom10';
                     tags[toPush].push({text: tuple[2]})
                  });
               }
               var features = [], frequencies = [], totalFeatures = 30;
               for (var id in cluster.rif) {
                  if (!cluster.rif.hasOwnProperty(id)) continue;
                  frequencies.push({f: Number(id), features: cluster.rif[id]});
               }
               frequencies.sort(function (a, b) {
                  return b.f - a.f;
               });
               frequencies.some(function (freqObj) {
                  freqObj.features.some(function (feature) {
                     features.push(feature);
                     return --totalFeatures <= 0;
                  });
                  return totalFeatures <= 0;
               });
               
               var toReturn = {
                  progress: (total - count) / total,
                  id: cluster.cnoprefix,
                  tags: tags,
                  labels: labels[request.params.collection] || [],
                  features: features,
               };
               
               response.status(200);
               response.header('Content-Type', 'application/json');
               response.json(toReturn);
            });
         });
      });
   });
});

app.put('/api/:collection/:id', function (request, response) {
   var data = {};
   for (var qid in request.query) {
      if (!request.query.hasOwnProperty(qid)) continue;
      data[qid] = request.query[qid];
   }
   for (var bid in request.body) {
      if (!request.body.hasOwnProperty(bid)) continue;
      data[bid] = request.body[bid];
   }
   
   console.log(data);
   
   MongoClient.connect('mongodb://' + config.mongodb.client_host + '/' + config.mongodb.db_name, function (err, db) {
      var toSet = {};
      toSet[labelId] = data[labelId];
      db.collection(request.params.collection).update({cnoprefix: request.params.id}, {$set: toSet}, {upsert: false}, function (err, result) {
         var toSend = {
            message: 'Thank you for your data!',
         };
         response.status(200);
         response.header('Content-Type', 'application/json');
         response.json(toSend);
      });
   });
});


app.get('/partials/empty', function (request, response) {
   response.status(200);
   response.send('');
});

app.get('/partials/classify/:collection', function (request, response) {
   response.status(200);
   response.render('partials/clusterPartial', {});
});

app.get('*', function (request, response) {
   var variables = {
      vars: {},
      title: 'Home',
      appName: 'classifyApp',
      toClient: {
         collections: [],
      },
   };
   variables.vars[env] = true;
   var exclude = ['system.indexes'];
   
   MongoClient.connect('mongodb://' + config.mongodb.client_host + '/' + config.mongodb.db_name, function (err, db) {
      db.collectionNames(function (err, names) {
         db.close();
         for (var i = names.length - 1; i >= 0; i--) {
            if (exclude.indexOf(names[i].name) >= 0) names.splice(i, 1);
         }
         variables.toClient.collections = names;
         
         response.render('index', variables);
      });
   });
});

app.listen(port);