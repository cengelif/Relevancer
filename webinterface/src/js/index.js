'use strict';


(function (angular, window, undefined) {
   var labelId = 'labels';
   
   var app = angular.module('classifyApp', [
      'ngRoute',
      'controllers',
      'services',
      'directives',
   ]);
   
   app.config(['$routeProvider', function ($routeProvider) {
      $routeProvider.
      when('/', {
         templateUrl: '/partials/empty',
         controller: 'NoCtrl'
      }).
      when('/classify/:collection', {
         templateUrl: function (urlAttr) {
            return '/partials/classify/' + urlAttr.collection;
         },
         controller: 'ClassifyCtrl'
      }).
      otherwise({
         redirectTo: '/'
      });
   }]);
   
   
   
   /*--- Controllers ---*/
   var controllers = angular.module('controllers', []);
   
   controllers.controller('NoCtrl', ['$scope', function ($scope) {}]);
   
   controllers.controller('MainCtrl', ['$scope', function ($scope) {
      if (window.serverdata.collections) $scope.collections = window.serverdata.collections;
   }]);
   
   controllers.controller('ClassifyCtrl', ['$scope', '$routeParams', 'Clusters', function ($scope, $routeParams, Clusters) {
      $scope.$parent.currentCollection = $routeParams.collection;
      
      function resetScope (event) {
         $scope.sendingData = false;
         $scope.classification = '';
         getClusters();
      }
      
      function getClusters () {
         // console.log($scope.currentCollection);
         $scope.cluster = Clusters.get({collection: $scope.currentCollection}, function (response) {
            $scope.sendingData = false;
         });
      }
      
      $scope.sendClassification = function () {
         if (!$scope.cluster.id) return;
         if ($scope.classification === '' || !Object.keys($scope.classification).length) {
            $scope.cluster.labels.forEach(function (label) {
               label.notFilled = true;
            });
            return;
         } else if (Object.keys($scope.classification).length !== $scope.cluster.labels.length) {
            $scope.cluster.labels.forEach(function (label) {
               label.notFilled = !$scope.classification.hasOwnProperty(label.id);
            });
            return;
         }
         
         $scope.sendingData = true;
         var toSend = {
            collection: $scope.currentCollection,
            id: $scope.cluster.id,
         };
         var data = {};
         data[labelId] = $scope.classification;
         
         console.log('Data', data);
         
         Clusters.classify(toSend, data, function (response) {
            console.log('Response', response);
         });
         
         $scope.classification = '';
         getClusters();
      };
      
      resetScope();
   }]);
   
   
   
   /*--- Services ---*/
   var services = angular.module('services', ['ngResource']);
   
   services.factory('Clusters', ['$resource', function ($resource) {
      
      return $resource('/api/:collection/:id?', {}, {
         get: {
            method: 'GET',
            params: {},
         },
         classify: {
            method: 'PUT',
            params: {},
         },
      });
   }]);
   
   
   
   /*--- Directives ---*/
   var directives = angular.module('directives', []);
   
   directives.directive('tooltip', [function () {
      return {
         restrict: 'AE',
         link: function($scope, elem, attr) {
            setTimeout(function () {
               elem.tooltip();
            }, 500);
         }
      };
   }]);
   
})(window.angular, window);
