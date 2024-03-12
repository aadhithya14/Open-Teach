/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction.DistanceReticles
{
    public class DistantInteractionLineVisual : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IDistanceInteractor))]
        private MonoBehaviour _distanceInteractor;
        public IDistanceInteractor DistanceInteractor { get; protected set; }

        [SerializeField]
        private float _visualOffset = 0.07f;
        public float VisualOffset
        {
            get
            {
                return _visualOffset;
            }
            set
            {
                _visualOffset = value;
            }
        }
        [SerializeField]
        private float _lineWidth = 0.02f;
        public float LineWidth
        {
            get
            {
                return _lineWidth;
            }
            set
            {
                _lineWidth = value;
            }
        }
        [SerializeField]
        private Color _color = Color.white;
        public Color Color
        {
            get
            {
                return _color;
            }
            set
            {
                _color = value;
            }
        }
        [SerializeField]
        private Material _lineMaterial;

        [SerializeField]
        private bool _visibleDuringNormal;

        private PolylineRenderer _polylineRenderer;

        private List<Vector4> _linePoints;
        private IReticleData _target;

        private const int LINE_POINTS = 20;
        private const float TARGETLESS_LENGTH = 0.5f;

        protected bool _started;
        private bool _shouldDrawLine;
        private DummyPointReticle _dummyTarget = new DummyPointReticle();

        private void Awake()
        {
            DistanceInteractor = _distanceInteractor as IDistanceInteractor;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(DistanceInteractor);
            Assert.IsNotNull(_lineMaterial);
            _linePoints = new List<Vector4>(new Vector4[LINE_POINTS]);
            _polylineRenderer = new PolylineRenderer(_lineMaterial);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                DistanceInteractor.WhenStateChanged += HandleStateChanged;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                DistanceInteractor.WhenStateChanged -= HandleStateChanged;
            }
        }

        protected virtual void Update()
        {
            if (_shouldDrawLine)
            {
                UpdateLine();
            }
        }

        private void OnDestroy()
        {
            _polylineRenderer.Cleanup();
        }

        private void HandleStateChanged(InteractorStateChangeArgs args)
        {
            switch (args.NewState)
            {
                case InteractorState.Normal:
                    if (args.PreviousState != InteractorState.Disabled)
                    {
                        InteractableUnset();
                    }

                    break;
                case InteractorState.Hover:
                    if (args.PreviousState == InteractorState.Normal)
                    {
                        InteractableSet(DistanceInteractor.Candidate as MonoBehaviour);
                    }
                    break;
            }

            if (args.NewState == InteractorState.Select
                || args.NewState == InteractorState.Disabled
                || args.PreviousState == InteractorState.Disabled)
            {
                _shouldDrawLine = false;
            }
            else if (args.NewState == InteractorState.Hover)
            {
                _shouldDrawLine = true;
            }
            else if (args.NewState == InteractorState.Normal)
            {
                _shouldDrawLine = _visibleDuringNormal;
            }
        }
        private void InteractableSet(MonoBehaviour interactable)
        {
            if (interactable == null)
            {
                return;
            }
            if (interactable.TryGetComponent(out IReticleData reticleData))
            {
                _target = reticleData;
            }
            else if (interactable is IDistanceInteractable)
            {
                _dummyTarget.Target = (interactable as IDistanceInteractable).RelativeTo;
                _target = _dummyTarget;
            }
        }

        private void InteractableUnset()
        {
            _target = null;
        }


        private void UpdateLine()
        {
            ConicalFrustum frustum = DistanceInteractor.PointerFrustum;
            Vector3 start = frustum.StartPoint + frustum.Direction * _visualOffset;
            Vector3 end = TargetHit(frustum);
            Vector3 middle = start + frustum.Direction * Vector3.Distance(start, end) * 0.5f;

            for (int i = 0; i < LINE_POINTS; i++)
            {
                float t = i / (LINE_POINTS - 1f);
                Vector4 point = EvaluateBezier(start, middle, end, t);
                point.w = _lineWidth;
                _linePoints[i] = point;
            }

            _polylineRenderer.SetLines(_linePoints, _color);
            _polylineRenderer.RenderLines();
        }

        private Vector3 TargetHit(ConicalFrustum frustum)
        {
            if (_target != null)
            {
                return _target.GetTargetHit(frustum);
            }

            return frustum.StartPoint + frustum.Direction * TARGETLESS_LENGTH;
        }

        private static Vector3 EvaluateBezier(Vector3 start, Vector3 middle, Vector3 end, float t)
        {
            t = Mathf.Clamp01(t);
            float oneMinusT = 1f - t;
            return (oneMinusT * oneMinusT * start)
                + (2f * oneMinusT * t * middle)
                + (t * t * end);
        }

        private class DummyPointReticle : IReticleData
        {
            public Transform Target { get; set; }

            public Vector3 GetTargetHit(ConicalFrustum frustum)
            {
                return Target.position;
            }
        }

        #region Inject

        public void InjectAllDistantInteractionLineVisual(IDistanceInteractor interactor, Material material)
        {
            InjectDistanceInteractor(interactor);
            InjectLineMaterial(material);
        }

        public void InjectDistanceInteractor(IDistanceInteractor interactor)
        {
            _distanceInteractor = interactor as MonoBehaviour;
            DistanceInteractor = interactor;
        }

        public void InjectLineMaterial(Material material)
        {
            _lineMaterial = material;
        }

        #endregion
    }
}
