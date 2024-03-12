/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace Oculus.Interaction
{
    /// <summary>
    /// This componenet makes it possible to connect IPointables in the
    /// inspector to Unity Events that are broadcast on IPointable events.
    /// </summary>
    public class PointableUnityEventWrapper : MonoBehaviour
    {
        [SerializeField, Interface(typeof(IPointable))]
        private MonoBehaviour _pointable;
        private IPointable Pointable;

        [SerializeField]
        private UnityEvent _whenHover;
        [SerializeField]
        private UnityEvent _whenUnhover;
        [SerializeField]
        private UnityEvent _whenSelect;
        [SerializeField]
        private UnityEvent _whenUnselect;
        [SerializeField]
        private UnityEvent _whenMove;
        [SerializeField]
        private UnityEvent _whenCancel;

        public UnityEvent WhenHover => _whenHover;
        public UnityEvent WhenUnhover => _whenUnhover;
        public UnityEvent WhenSelect => _whenSelect;
        public UnityEvent WhenUnselect => _whenUnselect;
        public UnityEvent WhenMove => _whenMove;
        public UnityEvent WhenCancel => _whenCancel;

        protected bool _started = false;

        protected virtual void Awake()
        {
            Pointable = _pointable as IPointable;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);
            Assert.IsNotNull(Pointable);
            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                Pointable.WhenPointerEventRaised += HandlePointerEventRaised;
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                Pointable.WhenPointerEventRaised -= HandlePointerEventRaised;
            }
        }

        private void HandlePointerEventRaised(PointerArgs args)
        {
            switch (args.PointerEvent)
            {
                case PointerEvent.Hover:
                    _whenHover.Invoke();
                    break;
                case PointerEvent.Unhover:
                    _whenUnhover.Invoke();
                    break;
                case PointerEvent.Select:
                    _whenSelect.Invoke();
                    break;
                case PointerEvent.Unselect:
                    _whenUnselect.Invoke();
                    break;
                case PointerEvent.Move:
                    _whenMove.Invoke();
                    break;
                case PointerEvent.Cancel:
                    _whenCancel.Invoke();
                    break;
            }
        }

        #region Inject

        public void InjectAllPointableUnityEventWrapper(IPointable pointable)
        {
            InjectPointable(pointable);
        }

        public void InjectPointable(IPointable pointable)
        {
            _pointable = pointable as MonoBehaviour;
            Pointable = pointable;
        }

        #endregion
    }
}
