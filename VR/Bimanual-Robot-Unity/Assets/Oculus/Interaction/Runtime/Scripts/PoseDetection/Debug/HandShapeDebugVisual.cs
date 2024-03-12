/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.Input;
using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.PoseDetection.Debug
{
    public class HandShapeDebugVisual : MonoBehaviour
    {
        [SerializeField]
        private ShapeRecognizerActiveState _shapeRecognizerActiveState;

        [SerializeField]
        private Renderer _target;

        [SerializeField]
        private Color _normalColor = Color.red;

        [SerializeField]
        private Color _activeColor = Color.green;

        [SerializeField]
        private GameObject _fingerFeatureDebugVisualPrefab;

        [SerializeField]
        private Transform _fingerFeatureParent;

        [SerializeField]
        private Vector3 _fingerSpacingVec = new Vector3(0.0f, -1.0f, 0.0f);
        [SerializeField]
        private Vector3 _fingerFeatureSpacingVec = new Vector3(1.0f, 0.0f, 0.0f);

        [SerializeField]
        private Vector3 _fingerFeatureDebugLocalScale = new Vector3(0.3f, 0.3f, 0.3f);

        [SerializeField]
        private TextMeshPro _targetText;

        private Material _material;
        private bool _lastActiveValue = false;

        protected virtual void Awake()
        {
            Assert.IsNotNull(_shapeRecognizerActiveState);
            Assert.IsNotNull(_target);
            Assert.IsNotNull(_fingerFeatureDebugVisualPrefab);
            Assert.IsNotNull(_targetText);
            _material = _target.material;

            _material.color = _lastActiveValue ? _activeColor : _normalColor;

            if (_fingerFeatureParent == null)
            {
                _fingerFeatureParent = transform;
            }
        }

        protected virtual void Start()
        {
            bool foundAspect = _shapeRecognizerActiveState.Hand.GetHandAspect(out FingerFeatureStateProvider stateProvider);
            Assert.IsTrue(foundAspect);

            Vector3 fingerOffset = Vector3.zero;

            var statesByFinger = AllFeatureStates()
                .GroupBy(s => s.Item1)
                .Select(group => new
                {
                    HandFinger = group.Key,
                    FingerFeatures = group.SelectMany(item => item.Item2)
                });
            foreach (var g in statesByFinger)
            {
                Vector3 fingerDebugFeatureTotalDisp = fingerOffset;
                foreach (var config in g.FingerFeatures)
                {
                    var fingerFeatureDebugVisInst = Instantiate(_fingerFeatureDebugVisualPrefab, _fingerFeatureParent);
                    var debugVisComp = fingerFeatureDebugVisInst.GetComponent<FingerFeatureDebugVisual>();

                    debugVisComp.Initialize(g.HandFinger, config, stateProvider);
                    var debugVisTransform = debugVisComp.transform;
                    debugVisTransform.localScale = _fingerFeatureDebugLocalScale;
                    debugVisTransform.localRotation = Quaternion.identity;
                    debugVisTransform.localPosition = fingerDebugFeatureTotalDisp;

                    fingerDebugFeatureTotalDisp += _fingerFeatureSpacingVec;
                }

                fingerOffset += _fingerSpacingVec;
            }

            string shapeNames = "";
            foreach (ShapeRecognizer shapeRecognizer in _shapeRecognizerActiveState.Shapes)
            {
                shapeNames += shapeRecognizer.ShapeName;
            }

            _targetText.text = $"{_shapeRecognizerActiveState.Handedness} Hand: {shapeNames} ";
        }

        private IEnumerable<ValueTuple<HandFinger, IReadOnlyList<ShapeRecognizer.FingerFeatureConfig>>> AllFeatureStates()
        {
            foreach (ShapeRecognizer shapeRecognizer in _shapeRecognizerActiveState.Shapes)
            {
                foreach (var handFingerConfigs in shapeRecognizer.GetFingerFeatureConfigs())
                {
                    yield return handFingerConfigs;
                }
            }
        }

        protected virtual void OnDestroy()
        {
            Destroy(_material);
        }

        protected virtual void Update()
        {
            bool isActive = _shapeRecognizerActiveState.Active;
            if (_lastActiveValue != isActive)
            {
                _material.color = isActive ? _activeColor : _normalColor;
                _lastActiveValue = isActive;
            }
        }
    }
}
