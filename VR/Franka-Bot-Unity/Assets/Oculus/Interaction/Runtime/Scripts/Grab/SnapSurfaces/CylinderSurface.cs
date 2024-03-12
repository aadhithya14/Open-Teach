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
using System;

namespace Oculus.Interaction.HandGrab.SnapSurfaces
{
    [Serializable]
    public class CylinderSurfaceData : ICloneable
    {
        public object Clone()
        {
            CylinderSurfaceData clone = new CylinderSurfaceData();
            clone.startPoint = this.startPoint;
            clone.endPoint = this.endPoint;
            clone.angle = this.angle;
            return clone;
        }

        public CylinderSurfaceData Mirror()
        {
            CylinderSurfaceData mirror = Clone() as CylinderSurfaceData;
            return mirror;
        }

        public Vector3 startPoint = new Vector3(0f, 0.1f, 0f);
        public Vector3 endPoint = new Vector3(0f, -0.1f, 0f);

        [Range(0f, 360f)]
        public float angle = 360f;
    }

    /// <summary>
    /// This type of surface defines a cylinder in which the grip pose is valid around an object.
    /// An angle can be used to constrain the cylinder and not use a full circle.
    /// The radius is automatically specified as the distance from the axis of the cylinder to the original grip position.
    /// </summary>
    [Serializable]
    public class CylinderSurface : MonoBehaviour, ISnapSurface
    {
        [SerializeField]
        protected CylinderSurfaceData _data = new CylinderSurfaceData();

        /// <summary>
        /// Getter for the data-only version of this surface. Used so it can be stored when created
        /// at Play-Mode.
        /// </summary>
        public CylinderSurfaceData Data
        {
            get
            {
                return _data;
            }
            set
            {
                _data = value;
            }
        }

        [SerializeField]
        private Transform _relativeTo;

        [SerializeField]
        private Transform _gripPoint;

        /// <summary>
        /// Transform to which the surface refers to.
        /// </summary>
        public Transform RelativeTo
        {
            get => _relativeTo;
            set => _relativeTo = value;
        }
        /// <summary>
        /// Valid point at which the hand can snap, typically the SnapPoint position itself.
        /// </summary>
        public Transform GripPoint
        {
            get => _gripPoint;
            set => _gripPoint = value;
        }

        /// <summary>
        /// Direction from the axis of the cylinder to the original grip position.
        /// </summary>
        public Vector3 StartAngleDir
        {
            get
            {
                if (this.GripPoint == null)
                {
                    return Vector3.forward;
                }
                return Vector3.ProjectOnPlane(GripPoint.transform.position - StartPoint, Direction).normalized;
            }
        }

        /// <summary>
        /// Direction from the axis of the cylinder to the maximum angle allowance.
        /// </summary>
        public Vector3 EndAngleDir
        {
            get
            {
                return Quaternion.AngleAxis(Angle, Direction) * StartAngleDir;
            }
        }

        /// <summary>
        /// Base cap of the cylinder, in world coordinates.
        /// </summary>
        public Vector3 StartPoint
        {
            get
            {
                if (this.RelativeTo != null)
                {
                    return this.RelativeTo.TransformPoint(_data.startPoint);
                }
                else
                {
                    return _data.startPoint;
                }
            }
            set
            {
                if (this.RelativeTo != null)
                {
                    _data.startPoint = this.RelativeTo.InverseTransformPoint(value);
                }
                else
                {
                    _data.startPoint = value;
                }
            }
        }

        /// <summary>
        /// End cap of the cylinder, in world coordinates.
        /// </summary>
        public Vector3 EndPoint
        {
            get
            {
                if (this.RelativeTo != null)
                {
                    return this.RelativeTo.TransformPoint(_data.endPoint);
                }
                else
                {
                    return _data.endPoint;
                }
            }
            set
            {
                if (this.RelativeTo != null)
                {
                    _data.endPoint = this.RelativeTo.InverseTransformPoint(value);
                }
                else
                {
                    _data.endPoint = value;
                }
            }
        }

        /// <summary>
        /// The maximum angle for the surface of the cylinder, starting from the original grip position.
        /// To invert the direction of the angle, swap the caps order.
        /// </summary>
        public float Angle
        {
            get
            {
                return _data.angle;
            }
            set
            {
                _data.angle = Mathf.Repeat(value, 360f);
            }
        }

        /// <summary>
        /// The generated radius of the cylinder.
        /// Represents the distance from the axis of the cylinder to the original grip position.
        /// </summary>
        public float Radius
        {
            get
            {
                if (this.GripPoint == null)
                {
                    return 0f;
                }
                Vector3 start = StartPoint;
                Vector3 projectedPoint = start + Vector3.Project(this.GripPoint.position - start, Direction);
                return Vector3.Distance(projectedPoint, this.GripPoint.position);
            }
        }

        /// <summary>
        /// The direction of the central axis of the cylinder.
        /// </summary>
        public Vector3 Direction
        {
            get
            {
                Vector3 dir = (EndPoint - StartPoint);
                if (dir.sqrMagnitude == 0f)
                {
                    return this.RelativeTo ? this.RelativeTo.up : Vector3.up;
                }
                return dir.normalized;
            }
        }

        private float Height
        {
            get
            {
                return (EndPoint - StartPoint).magnitude;
            }
        }

        /// <summary>
        /// The rotation of the central axis of the cylinder.
        /// </summary>
        private Quaternion Rotation
        {
            get
            {
                if (_data.startPoint == _data.endPoint)
                {
                    return Quaternion.LookRotation(Vector3.forward);
                }
                return Quaternion.LookRotation(StartAngleDir, Direction);
            }
        }

        #region editor events
        private void Reset()
        {
            _gripPoint = this.transform;
            if (this.TryGetComponent(out HandGrabPose grabPose))
            {
                _relativeTo = grabPose.RelativeTo;
            }
        }
        #endregion

        protected virtual void Start()
        {
            Assert.IsNotNull(_relativeTo);
            Assert.IsNotNull(_gripPoint);
            Assert.IsNotNull(_data);
        }

        public Pose MirrorPose(in Pose pose)
        {
            Vector3 normal = Quaternion.Inverse(this.RelativeTo.rotation) * StartAngleDir;
            Vector3 tangent = Quaternion.Inverse(this.RelativeTo.rotation) * Direction;

            return pose.MirrorPoseRotation(normal, tangent);
        }

        private Vector3 PointAltitude(Vector3 point)
        {
            Vector3 start = StartPoint;
            Vector3 projectedPoint = start + Vector3.Project(point - start, Direction);
            return projectedPoint;
        }


        public float CalculateBestPoseAtSurface(in Pose targetPose, in Pose reference, out Pose bestPose, in PoseMeasureParameters scoringModifier)
        {
            return SnapSurfaceHelper.CalculateBestPoseAtSurface(targetPose, reference, out bestPose,
                scoringModifier, MinimalTranslationPoseAtSurface, MinimalRotationPoseAtSurface);
        }

        public ISnapSurface CreateMirroredSurface(GameObject gameObject)
        {
            CylinderSurface surface = gameObject.AddComponent<CylinderSurface>();
            surface.Data = _data.Mirror();
            return surface;
        }

        public ISnapSurface CreateDuplicatedSurface(GameObject gameObject)
        {
            CylinderSurface surface = gameObject.AddComponent<CylinderSurface>();
            surface.Data = _data;
            return surface;
        }

        protected Vector3 NearestPointInSurface(Vector3 targetPosition)
        {
            Vector3 start = StartPoint;
            Vector3 dir = Direction;
            Vector3 projectedVector = Vector3.Project(targetPosition - start, dir);

            if (projectedVector.magnitude > Height)
            {
                projectedVector = projectedVector.normalized * Height;
            }
            if (Vector3.Dot(projectedVector, dir) < 0f)
            {
                projectedVector = Vector3.zero;
            }

            Vector3 projectedPoint = StartPoint + projectedVector;
            Vector3 targetDirection = Vector3.ProjectOnPlane((targetPosition - projectedPoint), dir).normalized;
            //clamp of the surface
            float desiredAngle = Mathf.Repeat(Vector3.SignedAngle(StartAngleDir, targetDirection, dir), 360f);
            if (desiredAngle > Angle)
            {
                if (Mathf.Abs(desiredAngle - Angle) >= Mathf.Abs(360f - desiredAngle))
                {
                    targetDirection = StartAngleDir;
                }
                else
                {
                    targetDirection = EndAngleDir;
                }
            }
            Vector3 surfacePoint = projectedPoint + targetDirection * Radius;
            return surfacePoint;
        }

        public bool CalculateBestPoseAtSurface(Ray targetRay, in Pose recordedPose, out Pose bestPose)
        {
            Vector3 lineToCylinder = StartPoint - targetRay.origin;

            float perpendiculiarity = Vector3.Dot(targetRay.direction, Direction);
            float rayToLineDiff = Vector3.Dot(lineToCylinder, targetRay.direction);
            float cylinderToLineDiff = Vector3.Dot(lineToCylinder, Direction);

            float determinant = 1f / (perpendiculiarity * perpendiculiarity - 1f);

            float lineOffset = (perpendiculiarity * cylinderToLineDiff - rayToLineDiff) * determinant;
            float cylinderOffset = (cylinderToLineDiff - perpendiculiarity * rayToLineDiff) * determinant;

            Vector3 pointInLine = targetRay.origin + targetRay.direction * lineOffset;
            Vector3 pointInCylinder = StartPoint + Direction * cylinderOffset;
            float distanceToSurface = Mathf.Max(Vector3.Distance(pointInCylinder, pointInLine) - Radius);
            if (distanceToSurface < Radius)
            {
                float adjustedDistance = Mathf.Sqrt(Radius * Radius - distanceToSurface * distanceToSurface);
                pointInLine -= targetRay.direction * adjustedDistance;
            }
            Vector3 surfacePoint = NearestPointInSurface(pointInLine);
            Pose desiredPose = new Pose(surfacePoint, recordedPose.rotation);
            bestPose = MinimalTranslationPoseAtSurface(desiredPose, recordedPose);

            return true;
        }

        protected Pose MinimalRotationPoseAtSurface(in Pose userPose, in Pose snapPose)
        {
            Vector3 desiredPos = userPose.position;
            Quaternion desiredRot = userPose.rotation;
            Quaternion baseRot = snapPose.rotation;
            Quaternion rotDif = (desiredRot) * Quaternion.Inverse(baseRot);
            Vector3 desiredDirection = (rotDif * Rotation) * Vector3.forward;
            Vector3 projectedDirection = Vector3.ProjectOnPlane(desiredDirection, Direction).normalized;
            Vector3 altitudePoint = PointAltitude(desiredPos);
            Vector3 surfacePoint = NearestPointInSurface(altitudePoint + projectedDirection * Radius);
            Quaternion surfaceRotation = CalculateRotationOffset(surfacePoint) * baseRot;
            return new Pose(surfacePoint, surfaceRotation);
        }

        protected Pose MinimalTranslationPoseAtSurface(in Pose userPose, in Pose snapPose)
        {
            Vector3 desiredPos = userPose.position;
            Quaternion baseRot = snapPose.rotation;

            Vector3 surfacePoint = NearestPointInSurface(desiredPos);
            Quaternion surfaceRotation = CalculateRotationOffset(surfacePoint) * baseRot;

            return new Pose(surfacePoint, surfaceRotation);
        }

        protected Quaternion CalculateRotationOffset(Vector3 surfacePoint)
        {
            Vector3 recordedDirection = Vector3.ProjectOnPlane(this.GripPoint.position - StartPoint, Direction);
            Vector3 desiredDirection = Vector3.ProjectOnPlane(surfacePoint - StartPoint, Direction);
            return Quaternion.FromToRotation(recordedDirection, desiredDirection);
        }

        #region Inject

        public void InjectAllCylinderSurface(CylinderSurfaceData data,
            Transform relativeTo, Transform gripPoint)
        {
            InjectData(data);
            InjectRelativeTo(relativeTo);
            InjectGripPoint(gripPoint);
        }

        public void InjectData(CylinderSurfaceData data)
        {
            _data = data;
        }

        public void InjectRelativeTo(Transform relativeTo)
        {
            _relativeTo = relativeTo;
        }

        public void InjectGripPoint(Transform gripPoint)
        {
            _gripPoint = gripPoint;
        }

        #endregion
    }
}
