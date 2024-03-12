/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Oculus.Interaction
{
    public class PointableElement : MonoBehaviour, IPointableElement
    {
        [SerializeField]
        private bool _transferOnSecondSelection;

        [SerializeField]
        private bool _addNewPointsToFront = false;

        [SerializeField, Interface(typeof(IPointableElement)), Optional]
        private MonoBehaviour _forwardElement;

        public IPointableElement ForwardElement { get; private set; }

        #region Properties
        public bool TransferOnSecondSelection
        {
            get
            {
                return _transferOnSecondSelection;
            }
            set
            {
                _transferOnSecondSelection = value;
            }
        }

        public bool AddNewPointsToFront
        {
            get
            {
                return _addNewPointsToFront;
            }
            set
            {
                _addNewPointsToFront = value;
            }
        }
        #endregion

        public event Action<PointerArgs> WhenPointerEventRaised = delegate { };

        public List<Pose> Points => _points;
        public int PointsCount => _points.Count;

        public List<Pose> SelectingPoints => _selectingPoints;
        public int SelectingPointsCount => _selectingPoints.Count;

        protected List<Pose> _points;
        protected List<int> _pointIds;

        protected List<Pose> _selectingPoints;
        protected List<int> _selectingPointIds;

        protected bool _started = false;

        protected virtual void Awake()
        {
            ForwardElement = _forwardElement as IPointableElement;
        }

        protected virtual void Start()
        {
            this.BeginStart(ref _started);

            if (_forwardElement)
            {
                Assert.IsNotNull(ForwardElement);
            }

            _points = new List<Pose>();
            _pointIds = new List<int>();

            _selectingPoints = new List<Pose>();
            _selectingPointIds = new List<int>();


            this.EndStart(ref _started);
        }

        protected virtual void OnEnable()
        {
            if (_started)
            {
                if (ForwardElement != null)
                {
                    ForwardElement.WhenPointerEventRaised += HandlePointerEventRaised;
                }
            }
        }

        protected virtual void OnDisable()
        {
            if (_started)
            {
                if (ForwardElement != null)
                {
                    ForwardElement.WhenPointerEventRaised -= HandlePointerEventRaised;
                }
            }
        }

        private void HandlePointerEventRaised(PointerArgs args)
        {
            if (args.PointerEvent == PointerEvent.Cancel)
            {
                ProcessPointerEvent(args);
            }
        }

        public virtual void ProcessPointerEvent(PointerArgs args)
        {
            switch (args.PointerEvent)
            {
                case PointerEvent.Hover:
                    Hover(args);
                    break;
                case PointerEvent.Unhover:
                    Unhover(args);
                    break;
                case PointerEvent.Move:
                    Move(args);
                    break;
                case PointerEvent.Select:
                    Select(args);
                    break;
                case PointerEvent.Unselect:
                    Unselect(args);
                    break;
                case PointerEvent.Cancel:
                    Cancel(args);
                    break;
            }
        }

        private void Hover(PointerArgs args)
        {
            if (_addNewPointsToFront)
            {
                _pointIds.Insert(0, args.Identifier);
                _points.Insert(0, args.Pose);
            }
            else
            {
                _pointIds.Add(args.Identifier);
                _points.Add(args.Pose);
            }

            PointableElementUpdated(args);
        }

        private void Move(PointerArgs args)
        {
            int index = _pointIds.IndexOf(args.Identifier);
            if (index == -1)
            {
                return;
            }
            _points[index] = args.Pose;

            index = _selectingPointIds.IndexOf(args.Identifier);
            if (index != -1)
            {
                _selectingPoints[index] = args.Pose;
            }

            PointableElementUpdated(args);
        }

        private void Unhover(PointerArgs args)
        {
            int index = _pointIds.IndexOf(args.Identifier);
            if (index == -1)
            {
                return;
            }

            _pointIds.RemoveAt(index);
            _points.RemoveAt(index);

            PointableElementUpdated(args);
        }

        private void Select(PointerArgs args)
        {
            if (_selectingPoints.Count == 1 && _transferOnSecondSelection)
            {
                Cancel(new PointerArgs(_selectingPointIds[0], PointerEvent.Cancel, _selectingPoints[0]));
            }

            if (_addNewPointsToFront)
            {
                _selectingPointIds.Insert(0, args.Identifier);
                _selectingPoints.Insert(0, args.Pose);
            }
            else
            {
                _selectingPointIds.Add(args.Identifier);
                _selectingPoints.Add(args.Pose);
            }

            PointableElementUpdated(args);
        }

        private void Unselect(PointerArgs args)
        {
            int index = _selectingPointIds.IndexOf(args.Identifier);
            if (index == -1)
            {
                return;
            }

            _selectingPointIds.RemoveAt(index);
            _selectingPoints.RemoveAt(index);

            PointableElementUpdated(args);
        }

        private void Cancel(PointerArgs args)
        {
            int index = _selectingPointIds.IndexOf(args.Identifier);
            if (index != -1)
            {
                _selectingPointIds.RemoveAt(index);
                _selectingPoints.RemoveAt(index);
            }

            index = _pointIds.IndexOf(args.Identifier);
            if (index != -1)
            {
                _pointIds.RemoveAt(index);
                _points.RemoveAt(index);
            }
            else
            {
                return;
            }

            PointableElementUpdated(args);
        }


        protected virtual void PointableElementUpdated(PointerArgs args)
        {
            if (ForwardElement != null)
            {
                ForwardElement.ProcessPointerEvent(args);
            }
            WhenPointerEventRaised.Invoke(args);
        }

        #region Inject

        public void InjectOptionalForwardElement(IPointableElement forwardElement)
        {
            ForwardElement = forwardElement;
            _forwardElement = forwardElement as MonoBehaviour;
        }

        #endregion
    }
}
